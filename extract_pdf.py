import sys
import os

def extract_text():
    pdf_path = "tesis.pdf"
    txt_path = "tesis_extracted.txt"
    
    print(f"Buscando {pdf_path}...")
    if not os.path.exists(pdf_path):
        print(f"❌ Error: No se encontró {pdf_path}")
        return
        
    extracted_text = ""
    success = False
    
    # Intentar con pypdf
    try:
        import pypdf
        print("📖 Intentando extracción con 'pypdf'...")
        reader = pypdf.PdfReader(pdf_path)
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                extracted_text += f"\n--- PAGINA {idx + 1} ---\n{text}\n"
        success = True
        print("✅ Extracción completada con 'pypdf'")
    except ImportError:
        print("ℹ️ 'pypdf' no está disponible.")
        
    # Intentar con PyPDF2 si pypdf falló o no está instalado
    if not success:
        try:
            import PyPDF2
            print("📖 Intentando extracción con 'PyPDF2'...")
            reader = PyPDF2.PdfReader(pdf_path)
            for idx, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    extracted_text += f"\n--- PAGINA {idx + 1} ---\n{text}\n"
            success = True
            print("✅ Extracción completada con 'PyPDF2'")
        except ImportError:
            print("ℹ️ 'PyPDF2' no está disponible.")
            
    # Intentar con fitz (PyMuPDF)
    if not success:
        try:
            import fitz
            print("📖 Intentando extracción con 'PyMuPDF' (fitz)...")
            doc = fitz.open(pdf_path)
            for idx, page in enumerate(doc):
                text = page.get_text()
                if text:
                    extracted_text += f"\n--- PAGINA {idx + 1} ---\n{text}\n"
            success = True
            print("✅ Extracción completada con 'PyMuPDF'")
        except ImportError:
            print("ℹ️ 'PyMuPDF' no está disponible.")

    # Si todo falla, intentar instalar pypdf
    if not success:
        print("⚠️ No se encontró ninguna librería de PDF instalada. Intentando importar pdfminer...")
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract
            print("📖 Intentando extracción con 'pdfminer'...")
            extracted_text = pdfminer_extract(pdf_path)
            success = True
            print("✅ Extracción completada con 'pdfminer'")
        except ImportError:
            print("❌ No hay librerías de PDF instaladas. Por favor instala pypdf corriendo: pip install pypdf")
            return
            
    if success and extracted_text.strip():
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        print(f"🎉 ¡Éxito! Texto extraído guardado en: {txt_path} ({len(extracted_text)} caracteres)")
    else:
        print("❌ No se pudo extraer texto o el PDF está vacío/escaneado como imagen.")

if __name__ == "__main__":
    extract_text()
