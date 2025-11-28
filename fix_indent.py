with open('workflow_visualizer.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    # Lines that had 20 spaces before need to be reduced to 16 spaces
    if line.startswith('                    ') and not line.startswith('                     '):
        # This has exactly 20 spaces - reduce to 16
        new_lines.append('                ' + line[20:])
    else:
        new_lines.append(line)

with open('workflow_visualizer.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f'Fixed {len([l for l in lines if l.startswith("                    ")])} lines')
