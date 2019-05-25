import csv

with open('C:\Users\lenovo\Desktop\scikit-feature-master\skfeature\data\CNAE-9.data.txt') as input_file:
   lines = input_file.readlines()
   newLines = []
   for line in lines:
      newLine = line.strip().split()
      newLines.append(int(newLine))

with open('CNAE-9.csv', 'wb') as test_file:
   file_writer = csv.writer(test_file)
   file_writer.writerows( newLines )