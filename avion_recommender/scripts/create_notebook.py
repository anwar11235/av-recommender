import nbformat as nbf
import re
import os

def create_notebook():
    # Create a new notebook
    nb = nbf.v4.new_notebook()
    
    # Read the Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, 'pipeline_test.py'), 'r') as f:
        content = f.read()
    
    # Split the content into cells based on comments and code
    cells = []
    current_cell = []
    in_multiline_string = False
    
    for line in content.split('\n'):
        # Handle multiline strings
        if '"""' in line:
            if not in_multiline_string:
                # Start of multiline string - create markdown cell
                if current_cell:
                    cells.append(('\n'.join(current_cell), 'code'))
                    current_cell = []
                current_cell = [line.replace('"""', '').strip()]
                in_multiline_string = True
            else:
                # End of multiline string
                cells.append(('\n'.join(current_cell), 'markdown'))
                current_cell = []
                in_multiline_string = False
            continue
            
        if in_multiline_string:
            current_cell.append(line)
            continue
            
        # Handle regular comments and code
        if line.startswith('# ') and not any(x in line for x in ['!git', '%cd', '!pip']):
            # New comment block - create markdown cell
            if current_cell:
                cells.append(('\n'.join(current_cell), 'code'))
                current_cell = []
            current_cell = [line.replace('# ', '')]
        elif line.startswith('#'):
            # Continue current cell
            current_cell.append(line)
        else:
            # Code line
            if current_cell and current_cell[0].startswith('# '):
                # Previous cell was markdown
                cells.append(('\n'.join(current_cell).replace('# ', ''), 'markdown'))
                current_cell = []
            current_cell.append(line)
    
    # Add the last cell
    if current_cell:
        cells.append(('\n'.join(current_cell), 'code'))
    
    # Create notebook cells
    for content, cell_type in cells:
        if not content.strip():
            continue
            
        if cell_type == 'markdown':
            nb.cells.append(nbf.v4.new_markdown_cell(content))
        else:
            nb.cells.append(nbf.v4.new_code_cell(content))
    
    # Write the notebook
    notebook_path = os.path.join(script_dir, '..', 'notebooks', 'pipeline_test.ipynb')
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_notebook() 