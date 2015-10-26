import sys
import pandas as pd
import argparse

def set_args(arguments):
  parser = argparse.ArgumentParser()
  parser.add_argument('--canvasFile', default="stat110.csv", help="Canvas CSV file.")
  parser.add_argument('--docFile', default="google_doc.csv", help="Exported Google Doc CSV File.")
  parser.add_argument('--namesFile', default="names.csv", help="CSV File mapping GoogleDoc Names to Canvas Names")
  parser.add_argument('--outputFile', default="results.csv", help="Output file for merged results.")
  args = parser.parse_args(arguments)
  return args

def main(arguments):
  args = set_args(arguments)

  print args

  # Load canvas file.
  gradefile = args.canvasFile
  grades = pd.read_csv(gradefile)

  # Drop top two rows (:points possible, NaN)
  grades = grades.drop([0,1],axis=0)

  # Load Google Doc file.
  google_doc = args.docFile
  google = pd.read_csv(google_doc)

  # Load names mappings from Name to Canvas Name
  names_map = args.namesFile
  names = pd.read_csv(names_map)

  # Join the names to the midterms, so then we can join the result to the Canvas data.
  googleNames = pd.merge(google, names, left_on="Name", right_on="Doc")

  # For some reason, 'HW2 electronic?' column comes in as 'object' type rather than float.
  # This is likely due to the fact that someone input Score(...) as a value for the column.
  # Therefore, we convert all of the HWX columns to float, in place.
  keys = range(2,6)
  for key in keys:
    googleNames["HW{} electronic?".format(key)] = googleNames['HW{} electronic?'.format(key)].convert_objects(convert_numeric=True)
  print "Finished converting all of the values!"

  # Merge Google.
  joinedResults = pd.merge(googleNames, grades, left_on="Canvas", right_on="Student")

  # Keep the important columns
  toKeep = ['Doc', 'M1', 'M2', 'M3', 'M4', 'M', 'Homework 1 (42101)', 'Homework 2 (46145)', 'Homework 3 (47478)', 'Homework 4 (48482)']
  toKeep += ['HW4 electronic?', 'HW3 electronic?', 'HW2 electronic?', 'Student']

  dataAll = joinedResults[toKeep]

  out = args.outputFile
  dataAll.to_csv(out)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
