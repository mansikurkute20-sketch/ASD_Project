from flask import Flask, render_template, request, session, make_response
from datetime import datetime
from io import BytesIO
from xml.sax.saxutils import escape
import pandas as pd
import joblib


app = Flask(__name__)
app.secret_key = 'asd-screening-report-key-change-before-deployment'


# ==================================================
# LOAD ALL 5 ML MODELS
# ==================================================

logistic_model = joblib.load(
    "logistic_regression_model.pkl"
)

knn_model = joblib.load(
    "knn_model.pkl"
)

naive_bayes_model = joblib.load(
    "naive_bayes_model.pkl"
)

decision_tree_model = joblib.load(
    "decision_tree_model.pkl"
)

random_forest_model = joblib.load(
    "random_forest_model.pkl"
)


# ==================================================
# LOAD SCALER AND FEATURE NAMES
# ==================================================

scaler = joblib.load(
    "scaler.pkl"
)

feature_names = joblib.load(
    "feature_names.pkl"
)


# ==================================================
# HOME PAGE
# ==================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==================================================
# PREDICTION
# ==================================================

@app.route("/predict", methods=["POST"])
def predict():

    child_name = request.form.get(
    "child_name", ""
    ).strip()

    if not child_name:
         return "Please enter child's name."

    try:

        # ==================================================
        # GET A1-A10 SCORES
        # ==================================================

        data = {}

        for i in range(1, 11):

            data[f"A{i}_Score"] = int(
                request.form[f"A{i}_Score"]
            )


        # ==================================================
        # GET BASIC INFORMATION
        # ==================================================

        data["age"] = float(
            request.form["age"]
        )

        data["gender"] = request.form["gender"]

        data["jundice"] = request.form["jundice"]

        data["austim"] = request.form["austim"]


        # ==================================================
        # GET ADDITIONAL INFORMATION
        # ==================================================

        data["ethnicity"] = request.form["ethnicity"]

        data["contry_of_res"] = request.form[
            "contry_of_res"
        ]

        data["used_app_before"] = request.form[
            "used_app_before"
        ]

        data["relation"] = request.form[
            "relation"
        ]


        # ==================================================
        # CREATE DATAFRAME
        # ==================================================

        input_df = pd.DataFrame([data])


        # ==================================================
        # CLEAN CATEGORICAL VALUES
        # ==================================================

        categorical_columns = [

            "gender",
            "ethnicity",
            "jundice",
            "austim",
            "contry_of_res",
            "used_app_before",
            "relation"

        ]


        for column in categorical_columns:

            input_df[column] = (
                input_df[column]
                .astype(str)
                .str.strip()
            )


        # ==================================================
        # FIX RELATION CAPITALIZATION
        # ==================================================

        input_df["relation"] = input_df[
            "relation"
        ].replace({

            "self": "Self"

        })


        # ==================================================
        # ONE-HOT ENCODING
        # ==================================================

        input_df = pd.get_dummies(

            input_df,

            columns=categorical_columns,

            drop_first=True,

            dtype=int

        )


        # ==================================================
        # MATCH TRAINING FEATURES
        # ==================================================

        input_df = input_df.reindex(

            columns=feature_names,

            fill_value=0

        )


        # ==================================================
        # SCALE INPUT
        # Used for Logistic Regression and KNN
        # ==================================================

        input_scaled = scaler.transform(
            input_df
        )


        # ==================================================
        # LOGISTIC REGRESSION
        # ==================================================

        logistic_prediction = (
            logistic_model
            .predict(input_scaled)[0]
        )

        logistic_probability = (
            logistic_model
            .predict_proba(input_scaled)[0][1]
        )


        # ==================================================
        # KNN
        # ==================================================

        knn_prediction = (
            knn_model
            .predict(input_scaled)[0]
        )

        knn_probability = (
            knn_model
            .predict_proba(input_scaled)[0][1]
        )


        # ==================================================
        # NAIVE BAYES
        # ==================================================

        naive_bayes_prediction = (
            naive_bayes_model
            .predict(input_df)[0]
        )

        naive_bayes_probability = (
            naive_bayes_model
            .predict_proba(input_df)[0][1]
        )


        # ==================================================
        # DECISION TREE
        # ==================================================

        decision_tree_prediction = (
            decision_tree_model
            .predict(input_df)[0]
        )

        decision_tree_probability = (
            decision_tree_model
            .predict_proba(input_df)[0][1]
        )


        # ==================================================
        # RANDOM FOREST
        # ==================================================

        random_forest_prediction = (
            random_forest_model
            .predict(input_df)[0]
        )

        random_forest_probability = (
            random_forest_model
            .predict_proba(input_df)[0][1]
        )


        # ==================================================
        # CONVERT PREDICTION TO TEXT
        # ==================================================

        def prediction_text(prediction):

            if prediction == 1:

                return "Positive Screening"

            else:

                return "Negative Screening"


        # ==================================================
        # CREATE MODEL RESULTS
        # ==================================================

        results = {

            "Logistic Regression": {

                "prediction":
                    prediction_text(
                        logistic_prediction
                    ),

                "positive_probability":
                    round(
                        logistic_probability * 100,
                        2
                    ),

                "negative_probability":
                    round(
                        (1 - logistic_probability) * 100,
                        2
                    ),

                "confidence":
                    round(
                        max(
                            logistic_probability,
                            1 - logistic_probability
                        ) * 100,
                        2
                    )
            },


            "KNN": {

                "prediction":
                    prediction_text(
                        knn_prediction
                    ),

                "positive_probability":
                    round(
                        knn_probability * 100,
                        2
                    ),

                "negative_probability":
                    round(
                        (1 - knn_probability) * 100,
                        2
                    ),

                "confidence":
                    round(
                        max(
                            knn_probability,
                            1 - knn_probability
                        ) * 100,
                        2
                    )
            },


            "Naive Bayes": {

                "prediction":
                    prediction_text(
                        naive_bayes_prediction
                    ),

                "positive_probability":
                    round(
                        naive_bayes_probability * 100,
                        2
                    ),

                "negative_probability":
                    round(
                        (1 - naive_bayes_probability) * 100,
                        2
                    ),

                "confidence":
                    round(
                        max(
                            naive_bayes_probability,
                            1 - naive_bayes_probability
                        ) * 100,
                        2
                    )
            },


            "Decision Tree": {

                "prediction":
                    prediction_text(
                        decision_tree_prediction
                    ),

                "positive_probability":
                    round(
                        decision_tree_probability * 100,
                        2
                    ),

                "negative_probability":
                    round(
                        (1 - decision_tree_probability) * 100,
                        2
                    ),

                "confidence":
                    round(
                        max(
                            decision_tree_probability,
                            1 - decision_tree_probability
                        ) * 100,
                        2
                    )
            },


            "Random Forest": {

                "prediction":
                    prediction_text(
                        random_forest_prediction
                    ),

                "positive_probability":
                    round(
                        random_forest_probability * 100,
                        2
                    ),

                "negative_probability":
                    round(
                        (1 - random_forest_probability) * 100,
                        2
                    ),

                "confidence":
                    round(
                        max(
                            random_forest_probability,
                            1 - random_forest_probability
                        ) * 100,
                        2
                    )
            }

        }


        # ==================================================
        # STORE ALL PREDICTIONS
        # ==================================================

        predictions = [

            logistic_prediction,

            knn_prediction,

            naive_bayes_prediction,

            decision_tree_prediction,

            random_forest_prediction

        ]


        # ==================================================
        # COUNT POSITIVE AND NEGATIVE PREDICTIONS
        # ==================================================

        positive_count = predictions.count(1)

        negative_count = predictions.count(0)


        # ==================================================
        # CALCULATE MODEL VOTING PERCENTAGE
        # ==================================================

        total_models = len(predictions)


        positive_percentage = round(

            (positive_count / total_models) * 100,

            2

        )


        negative_percentage = round(

            (negative_count / total_models) * 100,

            2

        )


        # ==================================================
        # MAJORITY VOTING RESULT
        # ==================================================

        if positive_count >= 3:

            final_result = (
                "Positive Screening"
            )

        else:

            final_result = (
                "Negative Screening"
            )

        # Save only the latest screening details for PDF download.
        question_texts = {
            "A1_Score": "Does your child look at you when you call his/her name?",
            "A2_Score": "How easy is it for you to get eye contact with your child?",
            "A3_Score": "Does your child point to indicate that he/she wants something?",
            "A4_Score": "Does your child point to share interest with you?",
            "A5_Score": "Does your child pretend? (For example, caring for dolls or talking on a toy phone)",
            "A6_Score": "Does your child follow where you are looking?",
            "A7_Score": "If you or someone in the family is visibly upset, does your child show signs of wanting to comfort them?",
            "A8_Score": "How would you describe your child's first words?",
            "A9_Score": "Does your child use simple gestures? (For example, waving goodbye)",
            "A10_Score": "Does your child stare at nothing with no apparent purpose?"
        }

        answer_options = {
            "A1_Score": {0: "Always / Usually", 1: "Sometimes / Rarely / Never"},
            "A2_Score": {0: "Very Easy / Quite Easy", 1: "Quite Difficult / Very Difficult / Impossible"},
            "A3_Score": {0: "Many times / A few times a day", 1: "A few times a week / Less than once a week / Never"},
            "A4_Score": {0: "Many times / A few times a day", 1: "A few times a week / Less than once a week / Never"},
            "A5_Score": {0: "Many times / A few times a day", 1: "A few times a week / Less than once a week / Never"},
            "A6_Score": {0: "Many times / A few times a day", 1: "A few times a week / Less than once a week / Never"},
            "A7_Score": {0: "Always / Usually", 1: "Sometimes / Rarely / Never"},
            "A8_Score": {0: "Very Typical / Quite Typical", 1: "Slightly Unusual / Very Unusual / No Words"},
            "A9_Score": {0: "Many times / A few times a day", 1: "A few times a week / Less than once a week / Never"},
            "A10_Score": {1: "Many times / A few times a day / A few times a week", 0: "Less than once a week / Never"}
        }

        saved_answers = []
        for key, question in question_texts.items():
            score = int(data[key])
            saved_answers.append({
                "question": question,
                "answer": answer_options[key][score],
                "score": score
            })

        session["screening_report"] = {
            "child_name": child_name,
            "date": datetime.now().strftime("%d-%m-%Y %I:%M %p"),
            "age": str(data["age"]),
            "gender": "Male" if data["gender"].lower() == "m" else "Female",
            "jundice": data["jundice"].title(),
            "family_history": data["austim"].title(),
            "ethnicity": data["ethnicity"],
            "country": data["contry_of_res"],
            "used_app_before": data["used_app_before"].title(),
            "relation": data["relation"],
            "answers": saved_answers,
            "score_one_count": sum(item["score"] == 1 for item in saved_answers),
            "score_zero_count": sum(item["score"] == 0 for item in saved_answers),
            "final_result": final_result
        }

        # ==================================================
        # SEND RESULTS TO RESULT PAGE
        # ==================================================

        return render_template(

            "result.html",

            child_name=child_name,

            final_result=final_result,

            results=results,

            positive_count=positive_count,

            negative_count=negative_count,

            positive_percentage=positive_percentage,

            negative_percentage=negative_percentage

        )


    # ==================================================
    # ERROR HANDLING
    # ==================================================

    except Exception as e:

        return f"""

        <h2>Error occurred</h2>

        <p>{str(e)}</p>

        <br>

        <a href="/">Go Back</a>

        """


# ==================================================
# RUN FLASK APPLICATION
# ==================================================



# ==================================================
# DOWNLOAD SCREENING REPORT AS PDF
# ==================================================

@app.route("/download-report")
def download_report():
    report = session.get("screening_report")

    if not report:
        return "No screening report found. Please complete the screening first.", 400

    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
        )
        from xml.sax.saxutils import escape

        buffer = BytesIO()

        document = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=17 * mm,
            bottomMargin=17 * mm
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#315c9b"),
            spaceAfter=5
        )

        subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#64748b"),
            spaceAfter=14
        )

        heading_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#315c9b"),
            spaceBefore=10,
            spaceAfter=6
        )

        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            spaceAfter=4
        )

        small_style = ParagraphStyle(
            "SmallText",
            parent=body_style,
            fontSize=8,
            leading=11
        )

        # Get Child's Name safely
        child_name = escape(str(report.get("child_name") or "Not provided"))

        story = [
            Paragraph("ASD SCREENING REPORT", title_style),

            Paragraph(
                "Autism Spectrum Disorder Screening Using Machine Learning",
                subtitle_style
            ),

            Paragraph("1. User Information", heading_style)
        ]

        # User Information including Child's Name
        info_rows = [
            [
                "Child's Name", child_name,
                "Screening Date & Time", escape(str(report["date"]))
            ],
            [
                "Child's Age", escape(str(report["age"])),
                "Gender", escape(str(report["gender"]))
            ],
            [
                "Person Completing Form", escape(str(report["relation"])),
                "Jaundice History", escape(str(report["jundice"]))
            ],
            [
                "Family History of Autism", escape(str(report["family_history"])),
                "Ethnicity", escape(str(report["ethnicity"]))
            ],
            [
                "Country of Residence", escape(str(report["country"])),
                "Used Application Before", escape(str(report["used_app_before"]))
            ]
        ]

        info_table = Table(
            info_rows,
            colWidths=[39*mm, 47*mm, 39*mm, 47*mm]
        )

        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f7fb")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7dee8")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))

        story.extend([
            info_table,
            Paragraph("2. Overall Screening Result", heading_style)
        ])

        result_color = (
            colors.HexColor("#a45b00")
            if report["final_result"] == "Positive Screening"
            else colors.HexColor("#16834a")
        )

        result_table = Table(
            [[Paragraph(
                f'<font color="{result_color.hexval()}"><b>{escape(str(report["final_result"]))}</b></font>',
                ParagraphStyle(
                    "ResultText",
                    parent=body_style,
                    fontSize=15,
                    alignment=TA_CENTER
                )
            )]],
            colWidths=[172*mm]
        )

        result_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f7fb")),
            ("BOX", (0, 0), (-1, -1), 1, result_color),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ]))

        story.extend([
            result_table,
            Paragraph("3. Screening Questions and Answers", heading_style)
        ])

        # Screening Questions and Answers
        answer_rows = [
            ["No.", "Question", "Selected Answer"]
        ]

        for number, item in enumerate(report["answers"], start=1):
            answer_rows.append([
                str(number),
                Paragraph(escape(str(item["question"])), small_style),
                Paragraph(escape(str(item["answer"])), small_style)
            ])

        answer_table = Table(
            answer_rows,
            colWidths=[12*mm, 91*mm, 69*mm],
            repeatRows=1
        )

        answer_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#315c9b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7dee8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                colors.white,
                colors.HexColor("#f7f9fc")
            ]),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))

        story.extend([
            answer_table,
            Paragraph("4. Screening Response Summary", heading_style),

            Paragraph(
                f'Total responses with score 1: <b>{report["score_one_count"]}</b><br/>'
                f'Total responses with score 0: <b>{report["score_zero_count"]}</b>',
                body_style
            ),

            Paragraph(
                "These counts summarize the questionnaire responses only. "
                "They do not identify a single question as the cause of the final result.",
                small_style
            ),

            Paragraph("5. Result Explanation and Recommendations", heading_style)
        ])

        # Recommendations
        if report["final_result"] == "Positive Screening":

            story.append(Paragraph(
                "The combined screening system returned a Positive Screening result. "
                "This indicates that further professional evaluation may be helpful. "
                "The response summary above shows the selected answers; "
                "it does not mean that any one answer caused this result.",
                body_style
            ))

            story.append(Paragraph(
                "Recommendations:<br/>"
                "- Stay calm; a positive screening result does not confirm ASD.<br/>"
                "- Discuss your concerns with a qualified healthcare professional.<br/>"
                "- Consider consulting a Developmental Pediatrician or Child Psychiatrist.<br/>"
                "- Keep notes about communication, social interaction and behaviour for the appointment.",
                body_style
            ))

        else:

            story.append(Paragraph(
                "The combined screening system returned a Negative Screening result "
                "based on its selected model voting rule. "
                "If you still have concerns about communication, behaviour or development, "
                "consult a qualified healthcare professional.",
                body_style
            ))

        # Important Disclaimer
        story.extend([
            Paragraph("6. Important Disclaimer", heading_style),

            Paragraph(
                "This system provides an initial ASD-related screening indication using machine learning. "
                "This result is not a medical diagnosis and should not replace professional medical evaluation. "
                "Please consult a qualified healthcare professional if you have concerns.",
                body_style
            ),

            Spacer(1, 8),

            Paragraph(
                "Generated by ASD Screening System | Academic Project",
                small_style
            )
        ])

        # Generate PDF
        document.build(story)

        pdf_data = buffer.getvalue()
        buffer.close()

        response = make_response(pdf_data)

        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = (
            'attachment; filename="ASD_Screening_Report.pdf"'
        )

        return response

    except ImportError:
        return "PDF library is missing. Please install it using: pip install reportlab", 500

    except Exception as error:
        app.logger.exception("PDF report generation failed")
        return "Unable to generate the PDF report. Please check the Flask terminal for details.", 500


if __name__ == "__main__":

    app.run(
        debug=True
    )