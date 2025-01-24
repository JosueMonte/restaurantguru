import markdown_toc

# El archivo README.md que quieres procesar
input_file = 'README.md'

# Lee el contenido del archivo
with open(input_file, 'r', encoding='utf-8') as file:
    content = file.read()

# Genera el índice
toc = markdown_toc.build_toc(content)

# Inserta el índice en el contenido del archivo
updated_content = markdown_toc.insert_toc(content, toc)

# Escribe el contenido actualizado de nuevo al archivo
with open(input_file, 'w', encoding='utf-8') as file:
    file.write(updated_content)

print("Índice generado e insertado correctamente.")
