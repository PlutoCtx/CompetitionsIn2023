# @Version: python3.10
# @Time: 2023/4/15 15:54
# @Author: MaxBrooks
# @Email: 15905898514@163.com
# @File: parse_xml.py
# @Software: PyCharm
# @User: chent

import argparse
import re

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True,
              help='path to iBug 300-W data split XML file')
ap.add_argument('-t', '--output', required=True,
              help='path output data split XML file')
args = vars(ap.parse_args)

LANDMARKS = set(list(range(36, 48)))

PART = re.compile("part name='[0-9]+'")

print("[info] parsing data split XML file...")
rows = open(args['input']).read().strip().split('\n')
output = open(args['output'], 'w')

for row in rows:
    parts = re.findall(PART, row)

    if len(parts) == 0:
        output.write('{}\n'.format(row))

    else:
        attr = "name='"
        i = row.find(attr)
        j = row.find("'", i + len(attr) + 1)
        name = int(row[i + len(attr): j])

        if name in LANDMARKS:
            output.write("{}\n".format(row))

output.close()