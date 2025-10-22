
import sys

infile = 'para_with_hyperlink.jsonl'
outfile = 'para_with_hyperlink_small.jsonl'
N = 10  

if len(sys.argv) > 1:
    N = int(sys.argv[1])

with open(infile, 'r', encoding='utf-8') as fin, open(outfile, 'w', encoding='utf-8') as fout:
    for i, line in enumerate(fin):
        if i >= N:
            break
        fout.write(line)

print(f'✓ 已生成 {outfile}，共 {N} 行')
