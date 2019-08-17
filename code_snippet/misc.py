# MATH
np.ceil(5.7)        # return 6.0
np.floor(5.7)       # return 5.0
np.round(5.7)       # return 6.0
round(5.7)          # return 6
s.min()
s.max()

# PANDAS
pd.options.display.max_rows = len(df)	# default 60

# PLOTTING
plt.rcParams['figure.figsize'] = (16, 10)
plt.colormaps()		# return all possible cmap
plt.cm.Reds			# Reds cmap

# CHEATSHEET REGEX
\d any number
\D anything but number
\w any character
\W anything but character
.  any character except newline
\b whitespace
\. dot character
+  match 1 or more
*  match 0 or 1
?  match 0 or 1
^  start string
$  end string
{} expect 1-3, ex \d{3} expect 3 digit number, \d{1,3} expect 1-3 digit number
[] range, ex [A-Z] expect all capital letter, [abcd] expect letter a, b, c, d
|  either or, ex \d{1,4}|[A-z]{4,10} expect 1-4 digit number or 4-10 character word
\n newline
\s space
\t tab
\e escape
\r return
