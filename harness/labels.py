"""Hand-maintained label vocabularies for sources whose HF copies ship bare ints.

twitter-financial-news-topic stores labels as integers with no ClassLabel
names; this id order is from the dataset card and was spot-checked against
sample texts for all 20 ids.
"""
TWITTER_FIN_TOPICS = [
    "Analyst Update", "Fed or Central Banks", "Company or Product News",
    "Treasuries or Corporate Debt", "Dividend", "Earnings", "Energy or Oil",
    "Financials", "Currencies", "General News or Opinion",
    "Gold or Metals or Materials", "IPO", "Legal or Regulation",
    "M&A or Investments", "Macro", "Markets", "Politics", "Personnel Change",
    "Stock Commentary", "Stock Movement",
]
