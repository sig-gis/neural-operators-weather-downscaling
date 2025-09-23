import os
import glob
import zipfile
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--startyear',type=str,required=False,default='2020')
parser.add_argument('--endyear',type=str,required=False,default='2022')
parser.add_argument('--path',type=str,required=False,default='./datasets/era5_california/')
parser.add_argument('--dest',type=str,required=False,default='./datasets/era5_california/')

args = parser.parse_args()

path = args.path
destination = args.dest

start_year = int(args.startyear)
end_year = int(args.endyear)
years = [ str(start_year +i ) for i in range(end_year - start_year + 1)] 
months = ['01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12']


def extract_zip(zip_path,destination_path):
    # fname = zip_path.split('/')[-1][:-4] + '.nc'
    with zipfile.ZipFile(zip_path,'r') as zip_ref:
        zip_ref.extractall(destination_path)
    print(f'Successfully extracted file: {zip_path}')



for year in years:
    year_folder = os.path.join(destination,year)
    if not os.path.exists(year_folder):
        os.mkdir(year_folder)
    
    for month in months:
        archives = glob.glob(os.path.join(path,f'*{year}*{month}*.zip'))
        month_folder = os.path.join(year_folder,month)
        if not os.path.exists(month_folder):
            os.mkdir(month_folder)
        

        for archive in archives:
            extract_zip(archive,month_folder)