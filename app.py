import streamlit as st
import pandas as pd
import openai
import os
import smtplib # For sending emails
from email.mime.text import MIMEText # For creating email messages
from email.mime.multipart import MIMEMultipart # For creating email messages
from email.mime.application import MIMEApplication # For PDF attachments
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
Analyze the provided AWS Cost dataset (e.g., from Book1.xlsx - Sheet1.csv) to extract critical financial signals and insights for May. Focus on account spending dynamics, adherence to AOP, and significant cost variations by directly utilizing the provided daily average metrics. The insights should be actionable and relevant for financial leadership (CFO perspective).

**Key Columns to Use from Your Dataset:**
* **Account Identifier:** `Linked Account Name` (primary for reporting), `Linked Account ID`, `Account Number`
* **Contextual/Segmentation (Optional):** `P&L`, `Owner`, or any column representing 'Nature' or 'Category'.
* **AOP/Budget Data for May:** `AOP Daily average May`
* **Actual Spending Data for May:** `Actual Daily average May`
* **Previous Period Benchmark:** `Actual Daily average Last month MTD`
* **(Note on other columns):** The dataset also contains `AOP Daily average May Best` and `Actual Daily average Last full month day wise data...`. These can be used for supplementary context if relevant, or if specific follow-up questions address them.

**Key Areas of Inquiry & Insights (Focusing on May data, using provided daily averages):**

0.  **Overall Spending Trend (Absolute Daily Averages):**
    * Calculate the sum of the column `Actual Daily average May` across all accounts to get the "Total Overall Actual Daily Spend for May." NOT AOP. You'll be penalized for using that
    * Calculate the sum of `Actual Daily average Last month MTD` across all accounts to get the "Total Overall Actual Daily Spend for Last Month MTD."
    * Report these two absolute numbers and the difference between them.
    * Comment on the overall month-over-month trend in daily spending for the entire portfolio.

1.  **Top Spending Accounts (Based on May Daily Average):**
    * Using `Linked Account Name`, identify accounts with the highest `Actual Daily average May`.
    * Comment on the concentration of spend if any particular accounts dominate.
    * If a 'Nature' or 'Category' column (e.g., `P&L`) is available, identify if top spending accounts are concentrated in specific categories/natures.

2.  **Performance vs. AOP (Based on May Daily Averages):**
    * For each `Linked Account Name`, compare its `Actual Daily average May` directly with its `AOP Daily average May`.
    * Calculate the variance (Actual Daily Avg - AOP Daily Avg) in both absolute daily amount and percentage.
    * Highlight accounts with significant positive variances (daily average spend considerably below AOP daily average) and significant negative variances (daily average spend considerably above AOP daily average).
    * Explain what makes these variances "significant" from a CFO's perspective (e.g., large percentage deviation impacting budget adherence, high absolute daily overspend, consistent trend across multiple high-value accounts).
    * If a 'Nature' or 'Category' column is available, summarize AOP performance at this category level. Which categories are most over/under budget based on daily averages?

3.  **Month-over-Month Spending Dynamics (Comparison of Daily Averages):**
    * For each `Linked Account Name`, compare its `Actual Daily average May` with its `Actual Daily average Last month MTD`.
    * Pinpoint accounts exhibiting notable increases ("sudden hikes") or decreases in their daily spending average.
    * For these accounts, quantify the change (percentage and absolute daily average amount).
    * Explain the potential implications of these changes (e.g., escalating costs needing control, positive cost management trends, potential shifts in operational activity).
    * If a 'Nature' or 'Category' column is available, analyze month-over-month dynamics at this category level. Which categories are driving the most significant increases or decreases in daily spend?

4.  **Investigation of Spending Anomalies (Day-Wise Data for May, if available):**
    * For accounts identified with notable increases in daily average spend (from point 3):
        * If detailed day-wise actual spending data for May is available within the provided dataset (beyond the `Actual Daily average May` summary column), analyze this granular data.
        * Identify specific dates or periods within May that show unusual spending patterns or appear to be the primary drivers of the increased daily average.
        * Based on these daily trends (if data is available), suggest potential underlying reasons for the increased spending or specific areas that warrant immediate further investigation by the business.

**General Instruction for Category Analysis:**
* If a column representing 'Nature', 'Category', or similar (like the provided `P&L` or `Owner` column) is available in the dataset, actively utilize it to provide aggregated insights at this category level throughout your analysis. Highlight any categories that are major drivers of spending, AOP variance, or month-over-month changes.

**Guiding Principles for CFO-Level Insights:**
* **Materiality:** Focus on the most financially significant movements and variances.
* **Risk Identification:** Clearly flag areas of unexpected cost escalation, significant AOP deviations, or negative trends that could impact financial targets.
* **Opportunity Identification:** Highlight areas of effective cost management, positive AOP variances, or favorable trends.
* **Actionability:** Frame insights in a way that suggests potential actions or areas requiring deeper scrutiny.
* **Conciseness & Clarity:** Use clear business language.

**Output Format:**
Please present the findings in a structured manner, using tables for summaries and narrative explanations for insights and implications:

* **Overall Spending Trend (May vs. Last Month MTD - Daily Averages):**
    * Total Overall Actual Daily Spend for May: [`Calculated Sum`]
    * Total Overall Actual Daily Spend for Last Month MTD: [`Calculated Sum`]
    * Difference (May - Last Month MTD): [`Calculated Difference`]
    * Commentary on the overall trend.

* **Top Spending Accounts (May - Based on Daily Average):**
    * List of top accounts and their `Actual Daily average May`.
    * Brief commentary on spending concentration and category concentration (if applicable).

* **Performance vs. AOP (May - Based on Daily Averages) (Table & Insights):**
    | Linked Account Name | Category/Nature (if avail) | Actual Daily Avg May | AOP Daily Avg May | Daily Variance ($) | Daily Variance (%) | Key Observation/Implication |
    |---------------------|----------------------------|----------------------|-------------------|--------------------|--------------------|-----------------------------|
    | ...                 | ...                        | ...                  | ...               | ...                | ...                | ...                         |
    * Narrative highlighting the most significant over-performing and under-performing accounts and categories (if applicable) against AOP daily averages and why they matter.

* **Month-over-Month Spending Dynamics (Notable Changes in Daily Averages) (Table & Insights):**
    | Linked Account Name | Category/Nature (if avail) | Actual Daily Avg May | Actual Daily Avg Last Month MTD | Change in Daily Avg ($) | Change in Daily Avg (%) | Potential Implication/Concern |
    |---------------------|----------------------------|----------------------|---------------------------------|-------------------------|-------------------------|-------------------------------|
    | ...                 | ...                        | ...                  | ...                             | ...                     | ...                     | ...                           |
    * Narrative focusing on accounts and categories (if applicable) with the most substantial increases/decreases in daily average spend and the strategic importance of these shifts.

* **Deep Dive into Spending Anomalies (for each flagged account, if May daily data is available):**
    * **Account:** [`Linked Account Name`]
    * **Summary of May Daily Spending Trend:** [e.g., Consistent high spend, specific spike on ProjetP&L-MM-DD, increasing trend within the month]
    * **Key Dates/Periods of Concern in May:** [List dates/periods]
    * **Potential Reasons/Areas for Investigation:** [e.g., New campaign launch, unexpected vendor charges, increased resource usage, data entry error]
    * *(If detailed May daily data is not found, please state: "Detailed day-wise data for May not available for a deeper dive for this account based on provided columns.")*

**Important Notes for LLM:**
* Utilize the provided daily average columns (`Actual Daily average May`, `AOP Daily average May`, `Actual Daily average Last month MTD`) directly for all comparisons and insights. Do not perform MTD total calculations unless a specific MTD total column is explicitly provided in the dataset and its use is requested.
* When identifying "significant" variances or "notable" changes, use your analytical judgment based on the data patterns to highlight what would be of material interest or concern to financial leadership. Briefly explain the basis for your judgment (e.g., high percentage change relative to base, large absolute deviation, deviation from historical trends if discernible).
* Projections for the full month or other periods should **only** be provided if explicitly requested by the user in a follow-up query. The primary focus is on analyzing the data as presented.
* `P&L`, `Owner`, or other similar category/nature columns should be actively used to enrich the analysis by identifying if trends are concentrated within specific segments.
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


def send_email_with_pdf_attachment(subject, body_text, to_recipient, cc_recipient, pdf_bytes, pdf_filename="report.pdf"):
    """Sends an email with a PDF attachment using Gmail SMTP."""
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
    
    msg.attach(MIMEText(body_text, 'plain'))
    
    if pdf_bytes:
        part = MIMEApplication(pdf_bytes, Name=pdf_filename)
        part['Content-Disposition'] = f'attachment; filename="{pdf_filename}"'
        msg.attach(part)
    else:
        st.warning("No PDF content to attach. Email will be sent without PDF.")

    try:
        recipients_list = [to_recipient]
        if cc_recipient:
            recipients_list.append(cc_recipient)
        
        display_recipients = ", ".join(filter(None, [to_recipient, cc_recipient]))


        with st.spinner(f"📧 Sending email with PDF to {display_recipients}..."):
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender_email, sender_password)
                server.send_message(msg) # send_message handles To and Cc correctly
        st.success(f"📧 Email with PDF sent successfully to {display_recipients}!")
        return True
    except Exception as e:
        st.error(f"❌ Failed to send email with PDF: {e}")
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
    
    # Modified button text
    if st.button("📧 Email Initial Report (PDF) to Finance Team"):
        if st.session_state.initial_report_content:
            pdf_bytes = create_pdf_from_markdown_v2(st.session_state.initial_report_content) 
            if pdf_bytes:
                email_subject = f"AWS Analyzer Insights (PDF): {st.session_state.uploaded_file_name_for_subject}"
                email_body = "Please find the AWS analysis report attached as a PDF."
                # Sending to specific To and Cc recipients
                to_email = "manan.bedi@paytm.com"
                cc_email = "deepika.rawal@paytm.com"
                send_email_with_pdf_attachment(email_subject, email_body, to_email, cc_email, pdf_bytes, f"{st.session_state.uploaded_file_name_for_subject}_Report.pdf")
            else:
                st.error("Failed to generate PDF for email.")
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

