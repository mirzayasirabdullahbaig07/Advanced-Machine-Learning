"""
Handling Mixed Variables in a Dataset
--------------------------------------

What is mixed data?
- Mixed data means a single column contains both numerical and categorical values.
- Example: A 'Ticket' column in Titanic dataset, where some values are purely numbers (e.g., '12345')
  and some contain letters with numbers (e.g., 'PC 17599').

Approach:
1. Extract the numeric part (last split of the string).
2. Extract the categorical prefix (first split of the string).
3. Convert numerical parts into integers and treat categorical parts separately.
"""

import pandas as pd
import numpy as np

# Example DataFrame with mixed 'Ticket' values
df = pd.DataFrame({
    "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "349909"]
})

# -------------------------------
# Extract the last bit of ticket as number
# -------------------------------
df['ticket_num'] = df['Ticket'].apply(lambda s: s.split()[-1])   # get last token
df['ticket_num'] = pd.to_numeric(df['ticket_num'],               # convert to numeric
                                 errors='coerce',                # non-numeric -> NaN
                                 downcast='integer')             # keep as integer

# -------------------------------
# Extract the first part of ticket as category
# -------------------------------
df['ticket_cat'] = df['Ticket'].apply(lambda s: s.split()[0])    # get first token
df['ticket_cat'] = np.where(df['ticket_cat'].str.isdigit(),      # if numeric → replace with NaN
                            np.nan,
                            df['ticket_cat'])

# Show refined DataFrame
print(df.head(20))
