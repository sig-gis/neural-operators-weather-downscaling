import os
import glob
import zipfile
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--startyear',type=str,required=False,default='2016')
parser.add_argument('--endyear',type=str,required=False,default='2017')
parser.add_argument('--path',type=str,required=False,default='/home/rdemilt/mnt/eofm-benchmark/data/climatedownscaling')
parser.add_argument('--dest',type=str,required=False,default='/home/rdemilt/mnt/eofm-benchmark/data/climatedownscaling/era5')

args = parser.parse_args()

path = args.path
destination = args.dest

start_year = args.startyear
end_year = args.endyear
years = [ str(start_year +i ) for i in range(end_year - start_year + 1)] 


def extract_zip(zip_path,destination_path):
    fname = zip_path.split('/')[-1][:-4] + '.nc'
    with zipfile.ZipFile(zip_path,'r') as zip_ref:
        zip_ref.extractall(os.path.join(destination_path,fname))
    print(f'Successfully extracted file: {zip_path}')



for year in years:
    archives = glob.glob(os.path.join(path,f'*{year}*.zip'))
    year_folder = os.path.join(destination,year)
    if not os.path.exists(year_folder):
        os.mkdir(year_folder)

    for archive in archives:
        extract_zip(archive,year_folder)