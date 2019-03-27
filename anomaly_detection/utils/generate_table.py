import numpy as np

def printTable(val, col_val, row_val, func=None, col_order=None, row_order=None):
  if col_order is None:
    col_order = [i for i in range(len(col_val))]
  if row_order is None:
    row_order = [i for i in range(len(row_val))]
  if func is None:
    func = lambda x: True

  for i in range(val.shape[0]):
    if val[i,0,0] == 0:
      val = val[0:i]
      break
  mean = np.mean(val, axis=0)
  std = np.std(val, axis=0)

  # Find max.
  max_ind = np.unravel_index(mean.argmax(), mean.shape)

  def isMax(col_ind, row_ind):
    return col_ind == max_ind[0] and row_ind == max_ind[1]

  row_mean = np.mean(mean, axis=0)
  col_mean = np.mean(mean, axis=1)

  tab = ''
  for col_ind in col_order:
    tab += ' & \\pbox{20cm}{' + col_val[col_ind] + '}'
  tab += '\\\\ \\hline\\hline'
  print (tab)
  grey = True
  for row_ind in row_order:
    if func(row_val[row_ind]):
      tab = row_val[row_ind]
      for col_ind in col_order:
        tab += ' & '
        if isMax(col_ind, row_ind):
          tab += '\\textbf{'
        tab += '{:.2f}'.format(mean[col_ind, row_ind]*100) + '$\\pm$'
        tab += '{:.2f}'.format(std[col_ind, row_ind]*100)
        if isMax(col_ind, row_ind):
          tab += '}'
      # tab += '{:.2f}'.format(row_mean[row_ind]*100) + ' \\\\'
      tab += ' \\\\'
      if grey:
        tab += ' \\rowcolor{lightgray}'
        grey = False
      else:
        grey = True
      print(tab)
  print('\\hline\\hline')
  # tab = 'Avg'
  # for col_ind in col_order:
  #   tab += ' & {:.2f}'.format(col_mean[col_ind]*100)
  # tab += '\\\\'
  # print(tab)
