import pdfplumber

with pdfplumber.open("Transactions_history_Erste_Bank_Polska_SA_27-04-2026_12_30.pdf") as pdf:
    for i, page in enumerate(pdf.pages[:2], 1):
        print(f"\n===== PAGE {i} =====")
        text = page.extract_text(x_tolerance=3, y_tolerance=3)
        if text:
            for line in text.split("\n"):
                print(repr(line))