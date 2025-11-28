"""
Complete fix for workflow_visualizer.py
"""

with open('workflow_visualizer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed_lines = []
i = 0

while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    
    # Skip empty lines or comments
    if not stripped or stripped.startswith('#'):
        fixed_lines.append(line)
        i += 1
        continue
    
    # Get the previous non-empty line
    prev_indent = 0
    prev_line = ""
    j = -1
    for j in range(i-1, -1, -1):
        if lines[j].strip() and not lines[j].strip().startswith('#'):
            prev_stripped = lines[j].lstrip()
            prev_indent = len(lines[j]) - len(prev_stripped)
            prev_line = lines[j].strip()
            break
    
    # Calculate current indentation
    current_indent = len(line) - len(stripped)
    
    # Check if previous line requires an indented block
    requires_indent = False
    if j >= 0 and prev_line:
        prev_line_clean = prev_line.rstrip()
        if prev_line_clean.endswith(':'):
            # Check if it's a statement that requires indentation
            for keyword in ['if ', 'elif ', 'else:', 'for ', 'while ', 'with ', 'try:', 'except', 'finally:', 'def ', 'class ']:
                if prev_line_clean.startswith(keyword) or (' ' + keyword in prev_line_clean):
                    requires_indent = True
                    break
    
    # If indentation is required but current line has wrong indent, fix it
    if requires_indent and current_indent <= prev_indent:
        # Don't fix if it's a keyword that should be at same level
        if not any(stripped.startswith(kw) for kw in ['elif ', 'else:', 'except', 'finally:']):
            expected_indent = prev_indent + 4
            fixed_lines.append(' ' * expected_indent + stripped + '\n')
            i += 1
            continue
    
    # Otherwise keep the line as is
    fixed_lines.append(line)
    i += 1

# Write fixed content
with open('workflow_visualizer.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("Fixed all indentation issues")
