import mysql.connector
import csv

conn = mysql.connector.connect(
    host ="localhost",
    database = "companydb",
    user = "sanjula" ,
    password = "1234"
)

cursor = conn.cursor()



with open('company_dataset.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip the header row if it exists
    
    # Iterate over the rows in the CSV file
    for row in csv_reader:
        # Extract the values from the CSV row
        Refno = row[0]
        PrjCat = row[1]
        Year_End  = row[2]
        AudType  = row[3]
        WDA_FAc  = row[4]
        FA_Cost  = row[5]
        D_Expense = row[6]
        Cur_Assets = row[7]
        Stock_Invent = row[8]
        Cash_Bank = row[9]
        Sh_Cap_Adv = row[10]
        Reserves  = row[11]
        NC_Liab	 = row[12]
        Cur_Liab  = row[13]
        Turnover  = row[14]
        Pro_Bef_Tax  = row[15]
        Tax	 = row[16]
        Pro_Aft_Tax  = row[17]
        Tot_Liab  = row[18]
        Tot_Assets = row[19]

        
        # Define the SQL INSERT statement
        query = "INSERT INTO company_records (Refno, PrjCat, Year_End, AudType, WDA_FA, FA_Cost, D_Expense, Cur_Assets ,Stock_Invent ,Cash_Bank ,Sh_Cap_Adv ,Reserves,NC_Liab	,Cur_Liab ,Turnover ,Pro_Bef_Tax ,Tax	,Pro_Aft_Tax ,Tot_Liab ,Tot_Assets) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) " 
        
        # Execute the SQL statement with the extracted values
        cursor.execute(query, ( Refno ,PrjCat ,Year_End  ,AudType  ,WDA_FAc  ,FA_Cost  ,D_Expense ,Cur_Assets ,Stock_Invent ,Cash_Bank ,Sh_Cap_Adv ,Reserves  ,NC_Liab	 ,Cur_Liab  ,Turnover  ,Pro_Bef_Tax  ,Tax	 ,Pro_Aft_Tax, Tot_Liab  ,Tot_Assets ))  
    
# Commit the changes to the database
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()




conn.close()