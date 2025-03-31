import PyPDF2
import pdfplumber
from pathlib import Path
import re


def extract_text(input_path, output_dir):
    # Открываем PDF файл
    for file in Path(input_path).glob("*.pdf"):
        with open(file, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            full_text = ""

            # Читаем каждую страницу PDF
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    full_text += text

        # Регулярное выражение для поиска кириллических символов
        pattern = re.compile(r'[\u0400-\u04FF]+')

        # Находим все слова, написанные кириллицей
        words = pattern.findall(full_text)

        # Соединяем найденные кириллические слова в строку
        text = " ".join(words)

        # Сохраняем результат в TXT файл
        txt_file = output_dir / f"{file.stem}.txt"
        with open(txt_file, 'w', encoding='utf-8') as text_file:
            text_file.write(text)

        print(f"{pdf_file.name} -> {txt_file.name}")

def pdf_to_txt(input_dir, output_dir, clean=True):
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process all PDF files in input directory
    input_path = Path(input_dir)
    for pdf_file in input_path.glob("*.pdf"):
        try:
            with pdfplumber.open(pdf_file) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

            # Create output filename
            txt_file = output_path / f"{pdf_file.stem}.txt"

            # Save text
            with open(txt_file, "w", encoding="utf-8") as f:
                if clean:
                    text = clean_text(text)
                f.write(text)

            print(f"Converted: {pdf_file.name} -> {txt_file.name}")

        except Exception as e:
            print(f"Failed to convert {pdf_file.name}: {str(e)}")

def clean_text(text):
    # Keep: letters, numbers, whitespace, and basic punctuation (. , ! ?)
    # Add/remove characters in the regex pattern as needed
    cleaned = re.sub(r'[^a-zA-Z\s•\n]', '', text, flags=re.UNICODE)
    # Collapse multiple whitespace characters
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

if __name__ == "__main__":
    input_directory = "../materials"  # Update this
    output_directory = "../materials/converted"  # Update this

    extract_text(input_directory, output_directory)