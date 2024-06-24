import pandas as pd

def write_to_excel(lines, slopes):
    # Convert lists to DataFrames
    df_lines = pd.DataFrame(lines, columns=['frame_count', 'xl1', 'xl2', 'xr1', 'xr2'])
    df_slopes = pd.DataFrame(slopes, columns=['frame_count', 'slope', 'intercept'])

    # Write DataFrames to Excel
    with pd.ExcelWriter('output.xlsx') as writer:
        df_lines.to_excel(writer, sheet_name='Lines', index=False)
        df_slopes.to_excel(writer, sheet_name='Slopes', index=False)