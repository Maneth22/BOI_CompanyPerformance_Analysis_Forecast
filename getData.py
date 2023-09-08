import mysql.connector
import csv
import pandas as pd




def getDataByCompanyRef(ref):
    conn = mysql.connector.connect(
        host ="localhost",
        database = "companydb",
        user = "sanjula" ,
        password = "1234"
        )
    

    query = "SELECT * FROM company_records WHERE Refno = %s"
    

    cursor = conn.cursor()
    cursor.execute(query, (ref,)) 

    df = pd.DataFrame(cursor.fetchall(), columns=cursor.column_names)

    cursor.close()
    conn.close()

    df = df.set_index('Year_End')
    df = df.drop('Refno', axis=1)
    df = df.drop('AudType', axis=1)
    df = df.drop('PrjCat', axis=1)
    df = df.drop('D_Expense', axis=1)

    return df

