import pandas as pd

def edit_file(file_path, edit_path):
    # Read the file lines
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Process lines to add a comma at the end of each line if needed
    updated_lines = []
    first_line = True
    for line in lines:
        # if first line
        if first_line:
            first_line = False
            updated_lines.append(line)
            continue
        # if line end with digit
        if line[-2].isdigit():
            line = line.rstrip() + '"\n'
            # remove all " in the line
            line = line.replace('"', '')
            updated_lines.append(line)
    
    # Write the updated lines back to the file
    with open(edit_path, 'w') as file:
        file.writelines(updated_lines)
    
    print(f"File '{edit_path}' has been updated.")

def merge_csv(file_path_list, merged_file_path):
    # merge the csv files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in file_path_list])
    # export to csv
    combined_csv.to_csv(merged_file_path, index=False, encoding='utf-8-sig')

# Specify the file path
file_path = '/home/bhchen/action_recognition/model/MultiView_Actions/data/NUMATest_CS_ori.csv'  # Replace with your file path
edit_path = '/home/bhchen/action_recognition/model/MultiView_Actions/data/NUMATest_CS.csv'  # Replace with your file path

# edit_file(file_path, edit_path)

# file_path = '/home/bhchen/action_recognition/model/MultiView_Actions/data/NUMATrain_CS_ori.csv'  # Replace with your file path
# edit_path = '/home/bhchen/action_recognition/model/MultiView_Actions/data/NUMATrain_CS.csv'  # Replace with your file path

file_path_list = ['/home/bhchen/action_recognition/model/MultiView_Actions/data/NUMATest_View{}.csv'.format(i) for i in range(1, 4)]
merged_file_path = '/home/bhchen/action_recognition/model/MultiView_Actions/data/NUMATest_AllView.csv'
merge_csv(file_path_list, merged_file_path)