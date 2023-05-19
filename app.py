import dash
import dash_table
from dash import dcc, html
from dash.dependencies import Input, Output
import base64
import io
import PyPDF2
import process
import pandas as pd
import recommender

# Создание экземпляра Dash
app = dash.Dash(__name__)

# Определение макета приложения
app.layout = html.Div(
    children=[
        html.H1("Приложение для рекомендации ИТ-вакансий"),
        dcc.Upload(
            id="upload-resume",
            children=html.Div(["Перетащите сюда PDF-файл или ", html.A("выберите файл")]),
            style={
                "width": "99%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=False,
        ),
        html.Div(id="output"),
        html.Div(
            id="table-output",
            style={"maxHeight": "400px", "overflowY": "scroll"},
        ),
    ]
)

# Callback-функция для обработки загруженного резюме
@app.callback(
    Output("output", "children"),
    Output("table-output", "children"),
    [Input("upload-resume", "contents")],
)
def process_resume(contents):
    if contents is not None:
        # # Преобразование содержимого в PDF-документ
        # _, content_string = contents.split(",")
        # decoded = base64.b64decode(content_string)
        # pdf_file = io.BytesIO(decoded)

        # # Чтение содержимого PDF-документа
        # pdf_reader = PyPDF2.PdfReader(pdf_file)
        # resume_text = ""
        # for page_num in range(len(pdf_reader.pages)):
        #     page = pdf_reader.pages[page_num]
        #     resume_text += page.extract_text()

        text = pdftotext(contents)
        resume_text = process.preprocess(text)
        resume_skills = process.extract_skills(resume_text)

        output = recommender.get_recommendations(resume_skills)
        # output.drop("Unnamed: 0", axis=1, inplace=True)

        output = output.astype(str)

        print(output)
        # Отображение содержимого резюме
        # return html.Div(
        #     children=[
        #         html.H3("Содержимое резюме:"),
        #         html.Pre(output),
        #     ]
        # )

        # Вывод DataFrame в виде таблицы
        table = dash_table.DataTable(
            columns=[{"name": col, "id": col} for col in output.columns],
            data=output.to_dict("records"),
            style_table={"maxHeight": "400px", "overflowY": "scroll"},
            style_cell={
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "maxWidth": 0,
            },
        )

        return "Резюме обработано", table

    return None, None

def pdftotext(contents):
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    pdf_file = io.BytesIO(decoded)

    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()

        text += page_text
    return text

# Запуск приложения
if __name__ == "__main__":
    app.run_server(debug=True)
