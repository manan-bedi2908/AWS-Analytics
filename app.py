import streamlit as st
import pandas as pd
import openai
import os
import smtplib # For sending emails
from email.mime.text import MIMEText # For creating email messages
from email.mime.multipart import MIMEMultipart # For creating email messages
from email.mime.application import MIMEApplication # For PDF attachments
from openai.error import OpenAIError
# For PDF Generation
import markdown2 # For converting markdown to HTML (pip install markdown2)
from xhtml2pdf import pisa # For converting HTML to PDF (pip install xhtml2pdf)
import io # For handling byte streams

# Azure OpenAI setup using Streamlit secrets
# Ensure these secrets are set in your Streamlit Cloud environment or local secrets.toml
try:
    openai.api_type = "azure"
    openai.api_key = st.secrets["AZURE_OPENAI_API_KEY"]
    openai.api_base = st.secrets["AZURE_API_BASE"]
    openai.api_version = st.secrets["AZURE_API_VERSION"]
    deployment_name = st.secrets["AZURE_DEPLOYMENT_NAME"]  # This is your deployment, not model name
except KeyError as e:
    st.error(f"Azure OpenAI secret not found: {e}. Please set it in your Streamlit secrets.")
    st.stop()
except Exception as e:
    st.error(f"Error initializing Azure OpenAI: {e}")
    st.stop()


# Streamlit app config
st.set_page_config(page_title="📊 AWS Cost Analyzer", layout="wide")
st.title("📊 AWS Cost Analyzer with Azure OpenAI")
st.markdown("Upload your AWS Cost file, and the system will automatically generate initial insights. You can then ask follow-up questions.")

# Initialize session state variables
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'file_data' not in st.session_state:
    st.session_state.file_data = {}
if 'initial_report_generated' not in st.session_state:
    st.session_state.initial_report_generated = False
if 'initial_report_content' not in st.session_state:
    st.session_state.initial_report_content = ""
if 'selected_sheets_for_report' not in st.session_state:
    st.session_state.selected_sheets_for_report = []
if 'uploaded_file_name_for_subject' not in st.session_state: # To store filename for email subject
    st.session_state.uploaded_file_name_for_subject = "AWS Cost Analysis"


# System prompt (as defined in the Canvas)
SYS_PROMPT = """
**Objective:**
Analyze the provided AWS Cost dataset to extract critical financial signals and insights for the current period (e.g., May). Focus on identifying material account spending dynamics, significant deviations from AOP (Annual Operating Plan), and notable cost variations compared to previous periods, by directly utilizing the provided daily average metrics. The insights must be actionable, highlight potential risks and opportunities, and be presented from a CFO's perspective.

**Key Columns to Use from Your Dataset (Ensure these exact names are present or adapt analysis if names differ slightly):**
* **Account Identifier:** `Linked Account Name` (primary for reporting), `Linked Account ID`, `Account Number` (use `Linked Account ID` if available, otherwise `Account Number` for ID purposes in tables).
* **Core Segmentation (Prioritize if available):** `Consumer/Merchant`, `Category`, `New Allocation(FY25) Direct / Common` (referred to as 'Allocation Type' in analysis), `Billed to Entity`
* **Contextual/Segmentation (Optional, use if core ones above are not present or for further detail):** `P&L`, `Owner`
* **AOP/Budget Data for Current Period (e.g., May):** `AOP Daily average May`
* **Actual Spending Data for Current Period (e.g., May):** `Actual Daily average May`
* **Previous Period Benchmark (Assumed to be April MTD for May analysis):** `Actual Daily average Last month MTD` (referred to as 'Actual Daily Avg Apr MTD' in tables)
* **(Note on other columns):** The dataset may also contain columns like `AOP Daily average May Best` and `Actual Daily average Last full month day wise data...`. These can be used for supplementary context if relevant, or if specific follow-up questions address them.

**Key Areas of Inquiry & Insights (Focusing on Current Period data, using provided daily averages):**

0.  **Overall Spending Trend (Absolute Daily Averages):**
    * Calculate the sum of `Actual Daily average May` across all accounts to get the "Overall Average Actual Daily Spend for May."
    * Calculate the sum of `Actual Daily average Last month MTD` across all accounts to get the "Overall Average Actual Daily Spend for Apr MTD."
    * Report these two absolute numbers and the difference between them (both absolute and percentage change).
    * Comment on the overall month-over-month trend in daily spending for the entire portfolio. Is the overall daily spend increasing or decreasing, and by how much?

1.  **Key Spending Drivers & Material Anomalies (Based on Current Period Daily Average):**
    * Using `Linked Account Name` and its corresponding ID (`Linked Account ID` or `Account Number`), identify accounts that are the most significant contributors (key drivers) to the `Actual Daily average May`. This is not limited to a fixed number of accounts; focus on materiality.
    * **Crucially, identify and highlight any anomalies in spending patterns across *all* accounts.** This includes accounts with historically low spend suddenly appearing high, disproportionate spend compared to their usual profile (if discernible from provided data or context), or any other unexpected deviations.
    * Comment on the concentration of spend: do a few accounts or categories disproportionately drive the total spend?
    * Analyze spending distribution by `Consumer/Merchant`, `Category`, `Allocation Type`, and `Billed to Entity`. Identify which segments are the largest contributors.

2.  **Performance vs. AOP (Critical Focus on Top 15 Overages - Based on Current Period Daily Averages):**
    * For each `Linked Account Name` (and its ID), compare its `Actual Daily average May` directly with its `AOP Daily average May`.
    * Calculate the variance as (`Actual Daily average May` - `AOP Daily average May`). This is the "Actual vs AOP" daily variance. A positive value for this variance indicates that actual spend exceeded AOP. Do not include accounts which are well below the AOP.
    * **Your primary focus must be to identify and thoroughly analyze accounts where this "Actual vs AOP" daily variance is positive and significant. List strictly the top 15 such accounts [Sorted by Daily Variance (%) in descending], sorted in descending order by the magnitude of this positive "Actual vs AOP" daily variance (i.e., the accounts that have exceeded AOP by the largest daily dollar amount should be listed first). When determining significance for this top 15 list, consider both the absolute daily dollar amount of the overspend and the percentage by which the AOP was exceeded.**
    * Explain what makes these overages "significant" from a CFO's perspective (e.g., high percentage deviation leading to budget overrun, large absolute daily overspend impacting cash flow, consistent overspending trend, risk to financial targets).
    * If a 'Nature' or 'Category' column is available, summarize AOP overage performance at this category level for these top overspending accounts.

3.  **Month-over-Month Spending Dynamics (Critical Focus on Sudden Hikes - Comparison of Daily Averages):**
    * For each `Linked Account Name` (and its ID), compare its `Actual Daily average May` with its `Actual Daily average Last month MTD`.
    * **Must be atleast 5. Your primary focus must be to identify and thoroughly analyze accounts exhibiting notable *increases* ("sudden hikes") in their daily spending average. When presenting these, sort them by the materiality of the hike, considering both absolute daily dollar increase and percentage increase. The most critical hikes should be highlighted first. This is not limited to a fixed number of accounts; focus on all material hikes.**
    * Also, identify significant decreases if they represent material cost savings or operational changes, and highlight these if they are particularly impactful.
    * For these accounts (especially those with hikes), quantify the change (percentage and absolute daily average amount).
    * Explain the potential implications of these changes from a CFO's viewpoint (e.g., escalating costs requiring immediate control, unexpected budget pressures, but also positive cost management trends if decreases are significant).
    * If a 'Nature' or 'Category' column is available, analyze month-over-month dynamics at this category level. Which categories are driving the most significant increases (especially sudden hikes) or decreases in daily spend?

4.  **Investigation of Spending Anomalies & Hikes (Day-Wise Data for Current Period, if available):**
    * For accounts identified with notable AOP exceedances (from point 2) or sudden hikes (from point 3):
        * If detailed day-wise actual spending data for the current period (e.g., May) is available within the provided dataset (beyond the `Actual Daily average May` summary column), analyze this granular data.
        * Identify specific dates or periods within the current month that show unusual spending patterns or appear to be the primary drivers of the AOP overage or the increased daily average.
        * Based on these daily trends (if data is available), suggest potential underlying reasons for the increased spending or specific areas that warrant immediate further investigation by the business.

**General Instruction for Segmentation Analysis:**
* Throughout your analysis (Spending Drivers, AOP Performance, MoM Dynamics), actively utilize `Consumer/Merchant`, `Category`, `New Allocation(FY25) Direct / Common` (as 'Allocation Type'), and `Billed to Entity` columns for aggregated insights. Highlight any segments that are major drivers of spending, AOP variance (especially overages), or month-over-month changes (especially hikes). Use `P&L` or `Owner` for further detail if these primary segmentations are insufficient or if specific patterns emerge there.

**Guiding Principles for CFO-Level Insights:**
* **Materiality & Significance:** Focus on the most financially significant movements, variances, and anomalies that would warrant a CFO's attention. Not all deviations are equally important.
* **Risk Identification:** Clearly flag areas of unexpected cost escalation, significant AOP deviations (especially overages), sudden hikes, and negative trends that could impact financial targets or indicate control weaknesses.
* **Opportunity Identification:** While focusing on risks, also highlight areas of effective cost management or significant positive AOP variances if they are material.
* **Actionability & Root Cause Thinking:** Frame insights in a way that suggests potential actions or areas requiring deeper scrutiny. Where possible, hypothesize potential root causes for anomalies.
* **Conciseness & Clarity:** Use clear, direct business language. Avoid jargon where possible. Summarize key takeaways effectively.
* **Data Adherence:** All insights and figures must be derived strictly from the provided data columns. Do not infer or create data points not present in the input.

**Output Format:**
Please present the findings in a structured manner, using tables for summaries and narrative explanations for insights and implications:
* Max date when the current month MTD file was updated
* **Overall Spending Trend (Current Period vs. Apr MTD - Daily Averages):**
    * Overall Average Actual Daily Spend for Current Period (e.g., May): [`Calculated Sum`]
    * Overall Average Actual Daily Spend for Apr MTD (from `Actual Daily average Last month MTD`): [`Calculated Sum`]
    * Difference (Current Period - Apr MTD): [`Calculated Difference $ and %`]
    * Commentary on the overall trend and its significance.

* **Key Spending Drivers & Material Anomalies (Current Period - Based on Daily Average):**
    * List of key accounts (include `Linked Account ID` or `Account Number`) with their `Actual Daily average May` and `Actual Daily average Last month MTD` (for Apr MTD context), noting why they are highlighted.
    * Brief commentary on spending concentration.
    * **Spending Breakdown by Key Segments (Tables & Insights):**
        * **By Consumer/Merchant:**
            | Consumer/Merchant Segment | Total Actual Daily Avg (May) ($) | Total Actual Daily Avg (Apr MTD) ($) | % of Total May Spend | Observation/Implication |
            |---------------------------|----------------------------------|--------------------------------------|--------------------|-------------------------|
            | ...                       | ...                              | ...                                  | ...                | ...                     |
            * Narrative highlighting top segments and concentration.
        * **By Category:**
            | Category Segment          | Total Actual Daily Avg (May) ($) | Total Actual Daily Avg (Apr MTD) ($) | % of Total May Spend | Observation/Implication |
            |---------------------------|----------------------------------|--------------------------------------|--------------------|-------------------------|
            | ...                       | ...                              | ...                                  | ...                | ...                     |
            * Narrative highlighting top segments and concentration.
        * **By Allocation Type (`New Allocation(FY25) Direct / Common`):**
            | Allocation Type           | Total Actual Daily Avg (May) ($) | Total Actual Daily Avg (Apr MTD) ($) | % of Total May Spend | Observation/Implication |
            |---------------------------|----------------------------------|--------------------------------------|--------------------|-------------------------|
            | ...                       | ...                              | ...                                  | ...                | ...                     |
            * Narrative highlighting distribution.
        * **By Billed to Entity:**
            | Billed to Entity Segment  | Total Actual Daily Avg (May) ($) | Total Actual Daily Avg (Apr MTD) ($) | % of Total May Spend | Observation/Implication |
            |---------------------------|----------------------------------|--------------------------------------|--------------------|-------------------------|
            | ...                       | ...                              | ...                                  | ...                | ...                     |
            * Narrative highlighting distribution.

* **Performance vs. AOP (Top 15 Significant Overages - Current Period Daily Averages) (Table & Insights) [Sort in descending order of percentages] (Daily Variance (%) must be positive and very few negative which are less than -10%):**
    | Linked Account ID / Account Number | Linked Account Name | Category/Nature (if avail) | Actual Daily Avg May | AOP Daily Avg May | Actual Daily Avg Apr MTD (for context) | Daily Variance ($) (Actual - AOP) | Daily Variance (%) (Actual - AOP) | Key Observation/Implication (Focus on Material Overages) |
    |------------------------------------|---------------------|----------------------------|----------------------|-------------------|----------------------------------------|-----------------------------------|-----------------------------------|-------------------------------------------------------|
    | ...                                | ...                 | ...                        | ...                  | ...               | ...                                    | ...                               | ...                               | ...                                                   |
    * Narrative highlighting the **top 15 accounts strictly** with the most *material overages* against AOP daily averages, **sorted by the "Daily Variance ($) (Actual - AOP)" column in descending order (higher positive values first)**. Explain the financial impact and potential risks.
    * **AOP Variance by Key Segments (Focus on Overages):**
        * **By Consumer/Merchant:** Table showing sum of Actual Daily Avg May vs. sum of AOP Daily Avg May per segment. Highlight segments with significant net overages.
        * **By Category:** Table showing sum of Actual Daily Avg May vs. sum of AOP Daily Avg May per segment. Highlight segments with significant net overages.
        * **By Allocation Type:** Table showing sum of Actual Daily Avg May vs. sum of AOP Daily Avg May per segment.
        * **By Billed to Entity:** Table showing sum of Actual Daily Avg May vs. sum of AOP Daily Avg May per segment.

* **Month-over-Month Spending Dynamics (Focus on Sudden Hikes & Material Changes - Daily Averages) (Table & Insights):**
    | Linked Account ID / Account Number | Linked Account Name | Category/Nature (if avail) | Actual Daily Avg May | Actual Daily Avg Apr MTD | Change in Daily Avg ($) | Change in Daily Avg (%) | Potential Implication/Concern (Focus on Hikes) |
    |------------------------------------|---------------------|----------------------------|----------------------|--------------------------|-------------------------|-------------------------|------------------------------------------------|
    | ...                                | ...                 | ...                        | ...                  | ...                      | ...                     | ...                     | ...                                            |
    * Narrative focusing on accounts and categories (if applicable) with the most substantial *increases (sudden hikes)* or material decreases in daily average spend, **sorted by the significance of the hike/change (considering both $ and %)**. Discuss the strategic importance and potential causes/risks.
    * **MoM Change by Key Segments (Focus on Hikes):**
        * **By Consumer/Merchant:** Table showing sum of `Actual Daily average May` vs. sum of `Actual Daily average Last month MTD` per segment. Highlight segments with significant net hikes.
        * **By Category:** Table showing sum of `Actual Daily average May` vs. sum of `Actual Daily average Last month MTD` per segment. Highlight segments with significant net hikes.
        * **By Allocation Type:** Table showing sum of `Actual Daily average May` vs. sum of `Actual Daily average Last month MTD` per segment.
        * **By Billed to Entity:** Table showing sum of `Actual Daily average May` vs. sum of `Actual Daily average Last month MTD` per segment.

* **Deep Dive into Spending Anomalies/Hikes (for each flagged account, if Current Period daily data is available):**
    * **Account:** [`Linked Account Name`] (ID: [`Linked Account ID / Account Number`])
    * **Summary of Current Period Daily Spending Trend:** [e.g., Consistent high spend, specific spike on ProjetP&L-MM-DD, increasing trend within the month]
    * **Key Dates/Periods of Concern in Current Period:** [List dates/periods]
    * **Potential Reasons/Areas for Investigation:** [e.g., New service launch, unexpected vendor charges, increased resource usage, data entry error, seasonal peak]
    * *(If detailed daily data for the current period is not found, please state: "Detailed day-wise data for the current period not available for a deeper dive for this account based on provided columns.")*

* **Final Executive Summary (Top 10 Key Insights/Action Items):**
    * 1. [Bullet point insight/action]
    * 2. [Bullet point insight/action]
    * ...
    * 10. [Bullet point insight/action]

**Important Notes for LLM:**
* Utilize the provided daily average columns (`Actual Daily average May`, `AOP Daily average May`, `Actual Daily average Last month MTD` which represents April MTD) directly for all comparisons and insights. Do not perform MTD total calculations unless a specific MTD total column is explicitly provided in the dataset and its use is requested.
* When identifying "significant" variances or "notable" changes/anomalies, use your analytical judgment based on the data patterns to highlight what would be of material interest or concern to financial leadership. Prioritize by impact (considering both absolute and percentage changes). Briefly explain the basis for your judgment.
* Projections for the full month or other periods should **only** be provided if explicitly requested by the user in a follow-up query. The primary focus is on analyzing the data as presented.
* `Consumer/Merchant`, `Category`, `New Allocation(FY25) Direct / Common` (as 'Allocation Type'), `Billed to Entity`, `P&L`, `Owner`, or other similar category/nature columns should be actively used to enrich the analysis by identifying if trends are concentrated within specific segments.
* **After providing the initial report, be prepared to answer follow-up questions from the user regarding the AWS cost data provided. You can elaborate on points from the report, perform different cuts of the data, or explore other aspects as long as it relates to the content of the uploaded file(s). Be open and flexible in addressing these subsequent queries.**
"""



def get_openai_response(user_query, data_context):
    """Generates a response from Azure OpenAI based on the user query and data context."""
    messages = [{"role": "system", "content": SYS_PROMPT}]
    messages.append({"role": "user", "content": f"Here is the AWS Cost data I am referring to:\n{data_context}\n\nMy question: {user_query}"})

    try:
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=messages,
            temperature=0.3,
            max_tokens=4000
        )
        return response["choices"][0]["message"]["content"]
    except openai.error.InvalidRequestError as e:
        if "maximum context length" in str(e):
            st.error(f"❌ Error: The request was too long and exceeded the model's token limit. Please try with fewer selected sheets, a shorter question, or contact support. Details: {e}")
        else:
            st.error(f"❌ Error communicating with Azure OpenAI (Invalid Request): {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error communicating with Azure OpenAI: {e}")
        return None

# This function is kept for PDF generation if needed elsewhere, but not used for the email body scenario.
def create_pdf_from_markdown_v2(markdown_content, filename="report.pdf"):
    """Creates a PDF file from markdown content using xhtml2pdf for better table support."""
    try:
        html_content = markdown2.markdown(markdown_content, extras=["tables", "fenced-code-blocks", "code-friendly"])

        html_with_style = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; font-size: 10pt; line-height: 1.6; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }}
                th, td {{ border: 1px solid #dddddd; text-align: left; padding: 8px; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                pre {{ background-color: #f5f5f5; border: 1px solid #ccc; padding: 10px; white-space: pre-wrap; word-wrap: break-word; }}
                code {{ font-family: monospace; }}
                h1 {{ font-size: 18pt; }}
                h2 {{ font-size: 16pt; }}
                h3 {{ font-size: 14pt; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        pdf_file = io.BytesIO()
        pisa_status = pisa.CreatePDF(
            io.StringIO(html_with_style),
            dest=pdf_file
        )

        if pisa_status.err:
            st.error(f"Error during PDF creation (pisa): {pisa_status.err}")
            return None

        pdf_file.seek(0)
        return pdf_file.getvalue()

    except Exception as e:
        st.error(f"Error generating PDF with xhtml2pdf: {e}")
        return None


# MODIFICATION: The function now expects body_text to be HTML if no PDF is attached.
# If pdf_bytes is None, body_text is assumed to be HTML.
# If pdf_bytes is provided, body_text is plain and PDF is an attachment.
def send_email_with_report_content(subject, body_content, to_recipient, cc_recipient, is_html_body=True, pdf_bytes=None, pdf_filename="report.pdf"):
    """Sends an email. If pdf_bytes is provided, it's an attachment and body_content is plain text.
       If pdf_bytes is None, body_content is the main email content (HTML or plain based on is_html_body)."""
    try:
        sender_email = st.secrets["GMAIL_SENDER_EMAIL"]
        sender_password = st.secrets["GMAIL_SENDER_APP_PASSWORD"]
    except KeyError as e:
        st.error(f"Email secret not found: {e}. Please set GMAIL_SENDER_EMAIL and GMAIL_SENDER_APP_PASSWORD in Streamlit secrets.")
        return False

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_recipient
    if cc_recipient:
        msg['Cc'] = cc_recipient
    msg['Subject'] = subject

    if is_html_body:
        msg.attach(MIMEText(body_content, 'html'))
    else:
        msg.attach(MIMEText(body_content, 'plain'))

    if pdf_bytes: # Attach PDF if provided
        part = MIMEApplication(pdf_bytes, Name=pdf_filename)
        part['Content-Disposition'] = f'attachment; filename="{pdf_filename}"'
        msg.attach(part)
    elif not is_html_body and not pdf_bytes : # Warning if sending plain text body without attachment (original simple case)
        st.info("Email sent with plain text body and no PDF attachment.")


    try:
        recipients_list = [to_recipient]
        if cc_recipient: # cc_recipient can be a comma-separated string of emails
            recipients_list.extend([email.strip() for email in cc_recipient.split(',')])

        display_recipients = ", ".join(filter(None, [to_recipient, cc_recipient]))

        with st.spinner(f"📧 Sending email to {display_recipients}..."):
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender_email, sender_password)
                server.send_message(msg) # send_message handles To and Cc correctly from headers
        st.success(f"📧 Email sent successfully to {display_recipients}!")
        return True
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")
        return False

# --- UI Layout ---
# Section 1: File Upload & Sheet Selection
st.subheader("📁 File Upload & Sheet Selection")
uploaded_file = st.file_uploader("Upload your AWS Cost file (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    if st.session_state.get('uploaded_file_name_for_subject') != uploaded_file.name:
        st.session_state.file_data = {}
        st.session_state.initial_report_generated = False
        st.session_state.initial_report_content = ""
        st.session_state.conversation_history = []
        st.session_state.selected_sheets_for_report = []
        st.session_state.uploaded_file_name_for_subject = uploaded_file.name
        st.info("New file detected. Session has been reset for new analysis.")

    if not st.session_state.file_data or st.session_state.uploaded_file_name_for_subject == uploaded_file.name:
        try:
            uploaded_file.seek(0)
            filename = uploaded_file.name.lower()
            current_file_data_temp = {}

            if filename.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                current_file_data_temp = {"Sheet1": df}
            elif filename.endswith(".xlsx"):
                xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
                current_file_data_temp = {
                    sheet_name: xls.parse(sheet_name)
                    for sheet_name in xls.sheet_names
                }
            else:
                raise ValueError("Unsupported file format.")

            st.session_state.file_data = current_file_data_temp
            st.success(f"✅ File '{uploaded_file.name}' loaded with {len(st.session_state.file_data)} sheet(s).")

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.session_state.file_data = {}
            st.session_state.initial_report_generated = False

if st.session_state.file_data:
    sheet_names = list(st.session_state.file_data.keys())

    if 'selected_sheets_multiselect' not in st.session_state:
        st.session_state.selected_sheets_multiselect = [sheet_names[0]] if sheet_names else []

    selected_sheets = st.multiselect(
        "📌 Select sheet(s) to analyze",
        options=sheet_names,
        default=st.session_state.selected_sheets_multiselect,
    )
    st.session_state.selected_sheets_multiselect = selected_sheets

    if selected_sheets:
        if selected_sheets != st.session_state.selected_sheets_for_report:
            st.session_state.selected_sheets_for_report = selected_sheets
            st.session_state.initial_report_generated = False
            st.session_state.initial_report_content = ""
            st.session_state.conversation_history = []
            st.info("Sheet selection changed. Initial report will regenerate. Conversation history reset.")

        for sheet in selected_sheets:
            with st.expander(f"📄 Preview: `{sheet}` (Top 10 rows)"):
                st.dataframe(st.session_state.file_data[sheet].head(10))

        if not st.session_state.initial_report_generated and st.session_state.selected_sheets_for_report:
            with st.spinner("🤖 Generating initial insights report... This may take a moment."):
                combined_text_for_initial_report = ""
                for sheet_name in st.session_state.selected_sheets_for_report:
                    if sheet_name in st.session_state.file_data:
                        df = st.session_state.file_data[sheet_name].head(1000)
                        df_text = df.to_string(index=False)
                        combined_text_for_initial_report += f"\n--- Data from Sheet: {sheet_name} ---\n{df_text}\n--- End of Sheet: {sheet_name} ---\n"
                    else:
                        st.warning(f"Sheet '{sheet_name}' not found in loaded data. Skipping for initial report.")

                if combined_text_for_initial_report:
                    initial_query = "Generate insights from the data based on the system prompt provided."
                    report_content = get_openai_response(initial_query, combined_text_for_initial_report)

                    if report_content:
                        st.session_state.initial_report_content = report_content
                        st.session_state.initial_report_generated = True
                        st.session_state.conversation_history.append({"role": "user", "content": initial_query + " (Automatic Initial Report)"})
                        st.session_state.conversation_history.append({"role": "assistant", "content": report_content})
                    else:
                        st.warning("Could not generate initial report. OpenAI call might have failed. Please try asking a question manually.")
                else:
                    st.warning("No data from selected sheets to generate initial report.")
st.divider() # Divider after the upload/selection section

# Section 2: AI Insights & Chat
st.subheader("💬 AI Insights & Chat")

if st.session_state.initial_report_generated and st.session_state.initial_report_content:
    st.markdown("### 🚀 Initial Insights Report")
    with st.container(height=400, border=True):
        st.markdown(st.session_state.initial_report_content)

    # MODIFICATION: Updated button text
    if st.button("📧 Email Initial Report (HTML Body) to Finance Team"):
        if st.session_state.initial_report_content:
            report_markdown = st.session_state.initial_report_content
            # Convert markdown to basic HTML for the email body
            html_report_part = markdown2.markdown(report_markdown, extras=["tables", "fenced-code-blocks", "code-friendly"])

            # Construct full HTML email body with styles (borrowed from PDF generation for consistency)
            email_html_body_content = f"""
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; font-size: 10pt; line-height: 1.6; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }} /* Email clients might handle width differently, 100% can be wide */
                    th, td {{ border: 1px solid #dddddd; text-align: left; padding: 8px; }}
                    th {{ background-color: #f2f2f2; font-weight: bold; }}
                    pre {{ background-color: #f5f5f5; border: 1px solid #ccc; padding: 10px; white-space: pre-wrap; word-wrap: break-word; }}
                    code {{ font-family: monospace; }}
                    h1 {{ font-size: 18pt; }}
                    h2 {{ font-size: 16pt; }}
                    h3 {{ font-size: 14pt; }}
                </style>
            </head>
            <body>
                {html_report_part}
            </body>
            </html>
            """

            email_subject = f"AWS Cost Summary"
            to_email = "manan.bedi@paytm.com"
            cc_email = "deepika.rawal@paytm.com,sharjeel@paytm.com"

            # MODIFICATION: Call the renamed/refactored email function
            # Pass the HTML content as the body, is_html_body=True, and pdf_bytes=None
            send_email_with_report_content(
                subject=email_subject,
                body_content=email_html_body_content,
                to_recipient=to_email,
                cc_recipient=cc_email,
                is_html_body=True,
                pdf_bytes=None # No PDF attachment
            )
        else:
            st.warning("No report content available to email.")
    st.divider()

if st.session_state.conversation_history:
    with st.expander("📜 View Conversation History", expanded=False):
        history_container_key = f"history_container_{len(st.session_state.conversation_history)}"
        with st.container(height=300, border=True):
            for i, msg in enumerate(st.session_state.conversation_history):
                role_emoji = "👤" if msg["role"] == "user" else "🤖"
                with st.chat_message(msg["role"], avatar=role_emoji):
                    st.markdown(msg["content"])
    st.divider()


if st.session_state.file_data and st.session_state.selected_sheets_for_report:
    user_question = st.chat_input("Ask a follow-up question about your AWS data...")

    if user_question:
        combined_text_for_chat = ""
        for sheet_name in st.session_state.selected_sheets_for_report:
                if sheet_name in st.session_state.file_data:
                    df = st.session_state.file_data[sheet_name].head(1000)
                    df_text = df.to_string(index=False)
                    combined_text_for_chat += f"\n--- Data from Sheet: {sheet_name} ---\n{df_text}\n--- End of Sheet: {sheet_name} ---\n"

        if not combined_text_for_chat:
            st.warning("No data from selected sheets to provide context for the question. Please select valid sheets.")
        else:
            st.session_state.conversation_history.append({"role": "user", "content": user_question})
            with st.spinner("🤔 Thinking..."):
                reply = get_openai_response(user_question, combined_text_for_chat)
                if reply:
                    st.session_state.conversation_history.append({"role": "assistant", "content": reply})
                else:
                    if st.session_state.conversation_history and st.session_state.conversation_history[-1]["role"] == "user":
                        st.session_state.conversation_history.pop()
                        st.warning("Failed to get a response from the AI. Your question was not processed.")
            st.rerun()
elif not st.session_state.file_data:
    st.info("☝️ Please upload a AWS Cost file to begin analysis.")
else:
    st.warning("Please select at least one sheet to enable chat and initial report generation.")