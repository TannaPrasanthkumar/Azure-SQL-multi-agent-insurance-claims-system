"""
Systematic fix for workflow_visualizer.py indentation issues
"""
import re

# Read the file
with open('workflow_visualizer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern 1: Fix lines after 'if' statements that should be indented
# This finds cases where after an if statement, the next line has less indentation
lines = content.split('\n')
fixed_lines = []
i = 0

while i < len(lines):
    current_line = lines[i]
    fixed_lines.append(current_line)
    
    # Check if this line is an if/elif/else/for/while/with statement
    stripped = current_line.lstrip()
    if stripped.startswith(('if ', 'elif ', 'else:', 'for ', 'while ', 'with ', 'try:', 'except', 'finally:', 'def ', 'class ')):
        # Calculate the expected indentation for the next line
        current_indent = len(current_line) - len(stripped)
        expected_indent = current_indent + 4
        
        # Check if there's a next line
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            next_stripped = next_line.lstrip()
            
            # Skip empty lines
            if next_stripped:
                next_indent = len(next_line) - len(next_stripped)
                
                # If next line is not indented enough, fix it
                if next_indent <= current_indent and not next_stripped.startswith(('#', 'else', 'elif', 'except', 'finally')):
                    # Re-indent the next line
                    lines[i + 1] = ' ' * expected_indent + next_stripped
    
    i += 1

# Write the fixed content
with open('workflow_visualizer.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(fixed_lines))

print("Fixed indentation issues in workflow_visualizer.py")
