# This notebook will be mainly used for the Capstone Project.

### Part 1: Peer-graded Assignment: Capstone Intro Notebook


```python
import pandas as pd
import numpy as np
```


```python
print("Hello Capstone Project Course!")
```

    Hello Capstone Project Course!


### Part 2: Peer-graded Assignment: Segmenting and Clustering Neighborhoods in Toronto

#### Setting up all the libraries


```python
!conda install -c conda-forge beautifulsoup4 --yes
!conda install -c conda-forge ProgressBar2 --yes
!conda install -c conda-forge lxml --yes
!conda install -c conda-forge geopy --yes
!conda install -c conda-forge folium=0.5.0 --yes
from progressbar import ProgressBar
from bs4 import BeautifulSoup as bts # library for web scraping
import numpy as np # library to handle data in a vectorized manner
import pandas as pd # library for data analysis
import matplotlib.cm as cm
import matplotlib.colors as colors
import requests # library to handle requests
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
import matplotlib as mp # library for visualization
import folium # map rendering library
import lxml
import re
from time import sleep
%matplotlib inline
```

    Solving environment: done
    
    
    ==> WARNING: A newer version of conda exists. <==
      current version: 4.5.11
      latest version: 4.7.11
    
    Please update conda by running
    
        $ conda update -n base -c defaults conda
    
    
    
    # All requested packages already installed.
    
    Solving environment: done
    
    
    ==> WARNING: A newer version of conda exists. <==
      current version: 4.5.11
      latest version: 4.7.11
    
    Please update conda by running
    
        $ conda update -n base -c defaults conda
    
    
    
    # All requested packages already installed.
    
    Solving environment: done
    
    
    ==> WARNING: A newer version of conda exists. <==
      current version: 4.5.11
      latest version: 4.7.11
    
    Please update conda by running
    
        $ conda update -n base -c defaults conda
    
    
    
    # All requested packages already installed.
    
    Solving environment: done
    
    
    ==> WARNING: A newer version of conda exists. <==
      current version: 4.5.11
      latest version: 4.7.11
    
    Please update conda by running
    
        $ conda update -n base -c defaults conda
    
    
    
    # All requested packages already installed.
    
    Solving environment: done
    
    
    ==> WARNING: A newer version of conda exists. <==
      current version: 4.5.11
      latest version: 4.7.11
    
    Please update conda by running
    
        $ conda update -n base -c defaults conda
    
    
    
    # All requested packages already installed.
    


#### Scrape the Toronto Wikipedia page and wrangle the data

#### Creating a beautifulsoup4 object from the Toronto Wikipedia page


```python
toronto_source  = requests.get('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M').text
```


```python
toronto_soup = bts(toronto_source,'lxml')
```

#### Find all tabulated data from the text source


```python
toronto_table = toronto_soup.findAll('table')
```

#### Removing citations and comments from scarped data


```python
toronto_table_text = toronto_table[0].tbody.text
toronto_table_text = re.sub("\[.*?\]", "", toronto_table_text)
toronto_table_text = re.sub("\(.*?\)", "", toronto_table_text)
```

#### The dataframe will consist of three columns: PostalCode, Borough, and Neighborhood


```python
# Splitting the string into a list at every new line, I have renamed the Postcode to PostalCode and Neighbourhood to Neighborhood as per the assignment
toronto_table_list = toronto_table_text.split('\n')
toronto_table_list[1] = 'PostalCode'
toronto_table_list[3] = 'Neighborhood'
del toronto_table_list[-1]
```

#### Creating the columns for the table


```python
toronto_table_columns = toronto_table_list[0:5]
```

#### Reshaping the data into a 2D NumPy array


```python
if (len(toronto_table_list[5:]) % 5 == 0):
    toronto_table_data = np.array(toronto_table_list[5:]).reshape(len(toronto_table_list[5:]) // 5,5)
else:
    print("Number of table elements is incorrect!")
```

#### Creating a Pandas DataFrame from the table


```python
toronto_dataframe =  pd.DataFrame(np.nan_to_num(toronto_table_data),columns = toronto_table_columns)
```


```python
# Displaying the top twelve rows of the DataFrame
print(toronto_dataframe.shape)
toronto_dataframe.head()
```

    (288, 5)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td></td>
      <td>M1A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
      <td></td>
    </tr>
    <tr>
      <td>1</td>
      <td></td>
      <td>M2A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
      <td></td>
    </tr>
    <tr>
      <td>2</td>
      <td></td>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
      <td></td>
    </tr>
    <tr>
      <td>3</td>
      <td></td>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
      <td></td>
    </tr>
    <tr>
      <td>4</td>
      <td></td>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



#### Dropping cells where Borough is equal to Not assigned

Only process the cells that have an assigned borough. Ignore cells with a borough that is Not assigned.


```python
toronto_dataframe.drop( toronto_dataframe[ toronto_dataframe['Borough'] == 'Not assigned' ].index , inplace=True)
print(toronto_dataframe.shape)
toronto_dataframe.head()
```

    (211, 5)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2</td>
      <td></td>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
      <td></td>
    </tr>
    <tr>
      <td>3</td>
      <td></td>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
      <td></td>
    </tr>
    <tr>
      <td>4</td>
      <td></td>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront</td>
      <td></td>
    </tr>
    <tr>
      <td>5</td>
      <td></td>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park</td>
      <td></td>
    </tr>
    <tr>
      <td>6</td>
      <td></td>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Heights</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



More than one neighborhood exist for one postal code area. For example, you will notice that M1B is listed twice and has two neighborhoods: Rouge and Malvern. These two rows is combined into one row with the neighborhoods separated with a comma as shown in first row in the below dataframe.


```python
toronto_dataframe = toronto_dataframe.groupby(['PostalCode','Borough'])['Neighborhood'].apply(lambda Neighborhood: ', '.join(Neighborhood)).to_frame()
toronto_dataframe.reset_index(inplace=True)
print(toronto_dataframe.shape)
toronto_dataframe.head()
```

    (103, 3)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Rouge, Malvern</td>
    </tr>
    <tr>
      <td>1</td>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Highland Creek, Rouge Hill, Port Union</td>
    </tr>
    <tr>
      <td>2</td>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
    </tr>
    <tr>
      <td>3</td>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
    </tr>
    <tr>
      <td>4</td>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
    </tr>
  </tbody>
</table>
</div>



#### Notice that M5A is listed twice and has two neighborhoods: Harbourfront and Regent Park. These two rows have been combined into one row with the neighborhoods separated with a comma. 


```python
print(toronto_dataframe.loc[toronto_dataframe['PostalCode'] == 'M5A'])
```

       PostalCode           Borough               Neighborhood
    53        M5A  Downtown Toronto  Harbourfront, Regent Park


#### If a cell has a borough but a Not assigned neighborhood, then the neighborhood will be the same as the borough

#### Notice for postalcode M7A the value of the Borough and the Neighborhood columns is Queen's Park.


```python
toronto_dataframe.loc[toronto_dataframe.Neighborhood == 'Not assigned', 'Neighborhood'] = toronto_dataframe.Borough
print(toronto_dataframe.loc[toronto_dataframe['PostalCode'] == 'M7A'])
```

       PostalCode       Borough  Neighborhood
    85        M7A  Queen's Park  Queen's Park


#### Using the .shape method to print the number of rows of your toronto_dataframe.


```python
toronto_dataframe.shape
```




    (103, 3)



Added two columns for populating Latitude and Longitude for each Postal Code


```python
toronto_dataframe['Latitude'] = np.nan
toronto_dataframe['Longitude'] = np.nan
toronto_dataframe.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Rouge, Malvern</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Highland Creek, Rouge Hill, Port Union</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
pbar = ProgressBar()
geolocator = Nominatim()
for index in pbar(range(0,toronto_dataframe['PostalCode'].shape[0])):
    address = toronto_dataframe.loc[index,'PostalCode'] + ", Ontario"
    location = geolocator.geocode(address, timeout = None)
    if (location != None):
        toronto_dataframe.loc[index,'Latitude'] = location.latitude
        toronto_dataframe.loc[index,'Longitude'] = location.longitude
    sleep(1)
print(toronto_dataframe.shape)
toronto_dataframe.head()
```

    /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/ipykernel_launcher.py:2: DeprecationWarning: Using Nominatim with the default "geopy/1.20.0" `user_agent` is strongly discouraged, as it violates Nominatim's ToS https://operations.osmfoundation.org/policies/nominatim/ and may possibly cause 403 and 429 HTTP errors. Please specify a custom `user_agent` with `Nominatim(user_agent="my-application")` or by overriding the default `user_agent`: `geopy.geocoders.options.default_user_agent = "my-application"`. In geopy 2.0 this will become an exception.
      
    N/A% (0 of 103) |                        | Elapsed Time: 0:00:00 ETA:  --:--:--/home/jupyterlab/conda/envs/python/lib/python3.6/site-packages/ipykernel_launcher.py:5: DeprecationWarning: `timeout=None` has been passed to a geocoder call. Using default geocoder timeout. In geopy 2.0 the behavior will be different: None will mean "no timeout" instead of "default geocoder timeout". Pass geopy.geocoders.base.DEFAULT_SENTINEL instead of None to get rid of this warning.
      """
    100% (103 of 103) |######################| Elapsed Time: 0:02:31 Time:  0:02:31


    (103, 5)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Rouge, Malvern</td>
      <td>34.065846</td>
      <td>-117.64843</td>
    </tr>
    <tr>
      <td>1</td>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Highland Creek, Rouge Hill, Port Union</td>
      <td>34.065846</td>
      <td>-117.64843</td>
    </tr>
    <tr>
      <td>2</td>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



#### If the above calls fails to populate the latitiude and longitude columns, I can still use the below csv file of geographical coordinates to populate those columns.


```python
file_name='http://cocl.us/Geospatial_data/Geospatial_Coordinates.csv'
tor_coordinates=pd.read_csv(file_name)
print(tor_coordinates.shape)
tor_coordinates.head()
```

    (103, 3)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>M1B</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <td>1</td>
      <td>M1C</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <td>2</td>
      <td>M1E</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <td>3</td>
      <td>M1G</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <td>4</td>
      <td>M1H</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
  </tbody>
</table>
</div>



### Populating the Latitude and Longitude columns from the tor_coordinates dataframe


```python
for index in range(0,toronto_dataframe['PostalCode'].shape[0]):
    postal_indexes = tor_coordinates.loc[tor_coordinates['Postal Code'] == toronto_dataframe['PostalCode'].iloc[index]]
    if (postal_indexes.index.size != 0):
        toronto_dataframe.loc[index, 'Latitude'] = tor_coordinates['Latitude'].iloc[postal_indexes.index[0]]
        toronto_dataframe.loc[index, 'Longitude'] = tor_coordinates['Longitude'].iloc[postal_indexes.index[0]]
```


```python
print(toronto_dataframe.shape)
toronto_dataframe.head()
```

    (103, 5)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Rouge, Malvern</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <td>1</td>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Highland Creek, Rouge Hill, Port Union</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <td>2</td>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <td>3</td>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <td>4</td>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
  </tbody>
</table>
</div>



#### Use geopy library to get the latitude and longitude values of Toronto City.


```python
address = 'Toronto, TO'

geolocator = Nominatim(user_agent="to_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of toronto are {}, {}.'.format(latitude, longitude))
```

    The geograpical coordinate of toronto are 43.6523873, -79.3835641.


## Explore and cluster the neighborhoods in Toronto

### All Toronto neighborhoods with boroughs that contain the word Toronto 

Slice the original dataframe toronto_dataframe and create a new dataframe tor_borough of all boroughs that contain the word Toronto.


```python
tor_borough = toronto_dataframe[toronto_dataframe['Borough'].str.contains('Toronto')].reset_index(drop=True)
print(tor_borough.shape)
tor_borough.head(38)
```

    (38, 5)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
    </tr>
    <tr>
      <td>1</td>
      <td>M4K</td>
      <td>East Toronto</td>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
    </tr>
    <tr>
      <td>2</td>
      <td>M4L</td>
      <td>East Toronto</td>
      <td>The Beaches West, India Bazaar</td>
      <td>43.668999</td>
      <td>-79.315572</td>
    </tr>
    <tr>
      <td>3</td>
      <td>M4M</td>
      <td>East Toronto</td>
      <td>Studio District</td>
      <td>43.659526</td>
      <td>-79.340923</td>
    </tr>
    <tr>
      <td>4</td>
      <td>M4N</td>
      <td>Central Toronto</td>
      <td>Lawrence Park</td>
      <td>43.728020</td>
      <td>-79.388790</td>
    </tr>
    <tr>
      <td>5</td>
      <td>M4P</td>
      <td>Central Toronto</td>
      <td>Davisville North</td>
      <td>43.712751</td>
      <td>-79.390197</td>
    </tr>
    <tr>
      <td>6</td>
      <td>M4R</td>
      <td>Central Toronto</td>
      <td>North Toronto West</td>
      <td>43.715383</td>
      <td>-79.405678</td>
    </tr>
    <tr>
      <td>7</td>
      <td>M4S</td>
      <td>Central Toronto</td>
      <td>Davisville</td>
      <td>43.704324</td>
      <td>-79.388790</td>
    </tr>
    <tr>
      <td>8</td>
      <td>M4T</td>
      <td>Central Toronto</td>
      <td>Moore Park, Summerhill East</td>
      <td>43.689574</td>
      <td>-79.383160</td>
    </tr>
    <tr>
      <td>9</td>
      <td>M4V</td>
      <td>Central Toronto</td>
      <td>Deer Park, Forest Hill SE, Rathnelly, South Hi...</td>
      <td>43.686412</td>
      <td>-79.400049</td>
    </tr>
    <tr>
      <td>10</td>
      <td>M4W</td>
      <td>Downtown Toronto</td>
      <td>Rosedale</td>
      <td>43.679563</td>
      <td>-79.377529</td>
    </tr>
    <tr>
      <td>11</td>
      <td>M4X</td>
      <td>Downtown Toronto</td>
      <td>Cabbagetown, St. James Town</td>
      <td>43.667967</td>
      <td>-79.367675</td>
    </tr>
    <tr>
      <td>12</td>
      <td>M4Y</td>
      <td>Downtown Toronto</td>
      <td>Church and Wellesley</td>
      <td>43.665860</td>
      <td>-79.383160</td>
    </tr>
    <tr>
      <td>13</td>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront, Regent Park</td>
      <td>43.654260</td>
      <td>-79.360636</td>
    </tr>
    <tr>
      <td>14</td>
      <td>M5B</td>
      <td>Downtown Toronto</td>
      <td>Ryerson, Garden District</td>
      <td>43.657162</td>
      <td>-79.378937</td>
    </tr>
    <tr>
      <td>15</td>
      <td>M5C</td>
      <td>Downtown Toronto</td>
      <td>St. James Town</td>
      <td>43.651494</td>
      <td>-79.375418</td>
    </tr>
    <tr>
      <td>16</td>
      <td>M5E</td>
      <td>Downtown Toronto</td>
      <td>Berczy Park</td>
      <td>43.644771</td>
      <td>-79.373306</td>
    </tr>
    <tr>
      <td>17</td>
      <td>M5G</td>
      <td>Downtown Toronto</td>
      <td>Central Bay Street</td>
      <td>43.657952</td>
      <td>-79.387383</td>
    </tr>
    <tr>
      <td>18</td>
      <td>M5H</td>
      <td>Downtown Toronto</td>
      <td>Adelaide, King, Richmond</td>
      <td>43.650571</td>
      <td>-79.384568</td>
    </tr>
    <tr>
      <td>19</td>
      <td>M5J</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront East, Toronto Islands, Union Station</td>
      <td>43.640816</td>
      <td>-79.381752</td>
    </tr>
    <tr>
      <td>20</td>
      <td>M5K</td>
      <td>Downtown Toronto</td>
      <td>Design Exchange, Toronto Dominion Centre</td>
      <td>43.647177</td>
      <td>-79.381576</td>
    </tr>
    <tr>
      <td>21</td>
      <td>M5L</td>
      <td>Downtown Toronto</td>
      <td>Commerce Court, Victoria Hotel</td>
      <td>43.648198</td>
      <td>-79.379817</td>
    </tr>
    <tr>
      <td>22</td>
      <td>M5N</td>
      <td>Central Toronto</td>
      <td>Roselawn</td>
      <td>43.711695</td>
      <td>-79.416936</td>
    </tr>
    <tr>
      <td>23</td>
      <td>M5P</td>
      <td>Central Toronto</td>
      <td>Forest Hill North, Forest Hill West</td>
      <td>43.696948</td>
      <td>-79.411307</td>
    </tr>
    <tr>
      <td>24</td>
      <td>M5R</td>
      <td>Central Toronto</td>
      <td>The Annex, North Midtown, Yorkville</td>
      <td>43.672710</td>
      <td>-79.405678</td>
    </tr>
    <tr>
      <td>25</td>
      <td>M5S</td>
      <td>Downtown Toronto</td>
      <td>Harbord, University of Toronto</td>
      <td>43.662696</td>
      <td>-79.400049</td>
    </tr>
    <tr>
      <td>26</td>
      <td>M5T</td>
      <td>Downtown Toronto</td>
      <td>Chinatown, Grange Park, Kensington Market</td>
      <td>43.653206</td>
      <td>-79.400049</td>
    </tr>
    <tr>
      <td>27</td>
      <td>M5V</td>
      <td>Downtown Toronto</td>
      <td>CN Tower, Bathurst Quay, Island airport, Harbo...</td>
      <td>43.628947</td>
      <td>-79.394420</td>
    </tr>
    <tr>
      <td>28</td>
      <td>M5W</td>
      <td>Downtown Toronto</td>
      <td>Stn A PO Boxes 25 The Esplanade</td>
      <td>43.646435</td>
      <td>-79.374846</td>
    </tr>
    <tr>
      <td>29</td>
      <td>M5X</td>
      <td>Downtown Toronto</td>
      <td>First Canadian Place, Underground city</td>
      <td>43.648429</td>
      <td>-79.382280</td>
    </tr>
    <tr>
      <td>30</td>
      <td>M6G</td>
      <td>Downtown Toronto</td>
      <td>Christie</td>
      <td>43.669542</td>
      <td>-79.422564</td>
    </tr>
    <tr>
      <td>31</td>
      <td>M6H</td>
      <td>West Toronto</td>
      <td>Dovercourt Village, Dufferin</td>
      <td>43.669005</td>
      <td>-79.442259</td>
    </tr>
    <tr>
      <td>32</td>
      <td>M6J</td>
      <td>West Toronto</td>
      <td>Little Portugal, Trinity</td>
      <td>43.647927</td>
      <td>-79.419750</td>
    </tr>
    <tr>
      <td>33</td>
      <td>M6K</td>
      <td>West Toronto</td>
      <td>Brockton, Exhibition Place, Parkdale Village</td>
      <td>43.636847</td>
      <td>-79.428191</td>
    </tr>
    <tr>
      <td>34</td>
      <td>M6P</td>
      <td>West Toronto</td>
      <td>High Park, The Junction South</td>
      <td>43.661608</td>
      <td>-79.464763</td>
    </tr>
    <tr>
      <td>35</td>
      <td>M6R</td>
      <td>West Toronto</td>
      <td>Parkdale, Roncesvalles</td>
      <td>43.648960</td>
      <td>-79.456325</td>
    </tr>
    <tr>
      <td>36</td>
      <td>M6S</td>
      <td>West Toronto</td>
      <td>Runnymede, Swansea</td>
      <td>43.651571</td>
      <td>-79.484450</td>
    </tr>
    <tr>
      <td>37</td>
      <td>M7Y</td>
      <td>East Toronto</td>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
    </tr>
  </tbody>
</table>
</div>



### Create a map of Toronto with neighborhoods superimposed on top where Boroughs that contain the word Toronto.


```python
# create map of New York using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, borough, neighborhood in zip(tor_borough['Latitude'], tor_borough['Longitude'], tor_borough['Borough'], tor_borough['Neighborhood']):
    label = '{}, {}'.format(borough, neighborhood)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfOGU0ZWUwOGMyZjVlNDRlMDliMmQxZmM5NzY2MzhhNDggewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzhlNGVlMDhjMmY1ZTQ0ZTA5YjJkMWZjOTc2NjM4YTQ4IiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF84ZTRlZTA4YzJmNWU0NGUwOWIyZDFmYzk3NjYzOGE0OCA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF84ZTRlZTA4YzJmNWU0NGUwOWIyZDFmYzk3NjYzOGE0OCcsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjUyMzg3MywtNzkuMzgzNTY0MV0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB6b29tOiAxMSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1heEJvdW5kczogYm91bmRzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbGF5ZXJzOiBbXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHdvcmxkQ29weUp1bXA6IGZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY3JzOiBMLkNSUy5FUFNHMzg1NwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9KTsKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHRpbGVfbGF5ZXJfMzRjYzMzZjExNWNhNDRlMTliODcxZjY0MzZlZjZlNzMgPSBMLnRpbGVMYXllcigKICAgICAgICAgICAgICAgICdodHRwczovL3tzfS50aWxlLm9wZW5zdHJlZXRtYXAub3JnL3t6fS97eH0ve3l9LnBuZycsCiAgICAgICAgICAgICAgICB7CiAgImF0dHJpYnV0aW9uIjogbnVsbCwKICAiZGV0ZWN0UmV0aW5hIjogZmFsc2UsCiAgIm1heFpvb20iOiAxOCwKICAibWluWm9vbSI6IDEsCiAgIm5vV3JhcCI6IGZhbHNlLAogICJzdWJkb21haW5zIjogImFiYyIKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfOGU0ZWUwOGMyZjVlNDRlMDliMmQxZmM5NzY2MzhhNDgpOwogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzAxODEyMWRhN2U5NDRhNjliYmQ4Mjg0ZTQ3N2JlZWQyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc2MzU3Mzk5OTk5OTksLTc5LjI5MzAzMTJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfOGU0ZWUwOGMyZjVlNDRlMDliMmQxZmM5NzY2MzhhNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOGJmMmJkNjVlMmE5NGJiMWI0M2QzMzBkOTJjODlhZTAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTQ1ODI3MDI1ZTg4NGEyNzljZGVjNjBjZGZmODZmNTEgPSAkKCc8ZGl2IGlkPSJodG1sX2U0NTgyNzAyNWU4ODRhMjc5Y2RlYzYwY2RmZjg2ZjUxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5FYXN0IFRvcm9udG8sIFRoZSBCZWFjaGVzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84YmYyYmQ2NWUyYTk0YmIxYjQzZDMzMGQ5MmM4OWFlMC5zZXRDb250ZW50KGh0bWxfZTQ1ODI3MDI1ZTg4NGEyNzljZGVjNjBjZGZmODZmNTEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDE4MTIxZGE3ZTk0NGE2OWJiZDgyODRlNDc3YmVlZDIuYmluZFBvcHVwKHBvcHVwXzhiZjJiZDY1ZTJhOTRiYjFiNDNkMzMwZDkyYzg5YWUwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzY3NTNjNDVkMjA4OTRlY2Q4ODg3YTZmMjIyODA2ODczID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc5NTU3MSwtNzkuMzUyMTg4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzhlNGVlMDhjMmY1ZTQ0ZTA5YjJkMWZjOTc2NjM4YTQ4KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzc0N2I4MTBhYWQ1NDQ2MzRhYWFjMGJkYzhhNDcxNzYzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E4MDhhNjU1YjFjZTQ0ODBhNTU0MWMzNTBjZWI0MTA1ID0gJCgnPGRpdiBpZD0iaHRtbF9hODA4YTY1NWIxY2U0NDgwYTU1NDFjMzUwY2ViNDEwNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RWFzdCBUb3JvbnRvLCBUaGUgRGFuZm9ydGggV2VzdCwgUml2ZXJkYWxlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83NDdiODEwYWFkNTQ0NjM0YWFhYzBiZGM4YTQ3MTc2My5zZXRDb250ZW50KGh0bWxfYTgwOGE2NTViMWNlNDQ4MGE1NTQxYzM1MGNlYjQxMDUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjc1M2M0NWQyMDg5NGVjZDg4ODdhNmYyMjI4MDY4NzMuYmluZFBvcHVwKHBvcHVwXzc0N2I4MTBhYWQ1NDQ2MzRhYWFjMGJkYzhhNDcxNzYzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2MxNjRjYTQzNTg1ZDQ0ZDZhMTkzNGY3NDEwZGYwZDE4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY4OTk4NSwtNzkuMzE1NTcxNTk5OTk5OThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfOGU0ZWUwOGMyZjVlNDRlMDliMmQxZmM5NzY2MzhhNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMmEzYjZmNTlhYjYzNGNjNjhjZTkyN2VkYTc1ZGVlZTcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTE2NTE0NWQ5ODE0NDcyYjlkZDgyZjJjMzMwMjM0YzMgPSAkKCc8ZGl2IGlkPSJodG1sX2UxNjUxNDVkOTgxNDQ3MmI5ZGQ4MmYyYzMzMDIzNGMzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5FYXN0IFRvcm9udG8sIFRoZSBCZWFjaGVzIFdlc3QsIEluZGlhIEJhemFhcjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMmEzYjZmNTlhYjYzNGNjNjhjZTkyN2VkYTc1ZGVlZTcuc2V0Q29udGVudChodG1sX2UxNjUxNDVkOTgxNDQ3MmI5ZGQ4MmYyYzMzMDIzNGMzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2MxNjRjYTQzNTg1ZDQ0ZDZhMTkzNGY3NDEwZGYwZDE4LmJpbmRQb3B1cChwb3B1cF8yYTNiNmY1OWFiNjM0Y2M2OGNlOTI3ZWRhNzVkZWVlNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mZjJmODMwNTNmZTE0OTM1OTkwOTAyMjMxNGM5NWQ5NCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1OTUyNTUsLTc5LjM0MDkyM10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84ZTRlZTA4YzJmNWU0NGUwOWIyZDFmYzk3NjYzOGE0OCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81ZmFkYmE2MzkxNmU0NTNkYjE0MjA5ZjZlY2Q2NzU4MCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zOTU2YWMwYjFmN2E0NDg2YjgwOGI0OGMyNmQzOWE0OCA9ICQoJzxkaXYgaWQ9Imh0bWxfMzk1NmFjMGIxZjdhNDQ4NmI4MDhiNDhjMjZkMzlhNDgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkVhc3QgVG9yb250bywgU3R1ZGlvIERpc3RyaWN0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81ZmFkYmE2MzkxNmU0NTNkYjE0MjA5ZjZlY2Q2NzU4MC5zZXRDb250ZW50KGh0bWxfMzk1NmFjMGIxZjdhNDQ4NmI4MDhiNDhjMjZkMzlhNDgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmYyZjgzMDUzZmUxNDkzNTk5MDkwMjIzMTRjOTVkOTQuYmluZFBvcHVwKHBvcHVwXzVmYWRiYTYzOTE2ZTQ1M2RiMTQyMDlmNmVjZDY3NTgwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzY4Y2I4YzIwZGFlZTRkZTE4ZWFhODQ5MTMyOTI3OWE1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI4MDIwNSwtNzkuMzg4NzkwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84ZTRlZTA4YzJmNWU0NGUwOWIyZDFmYzk3NjYzOGE0OCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zZjIxNTYzOGUxZGY0YTExYTY1NDcwMzIwZTg2NWM5MiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83OGJkNWU5Y2IzN2U0Yjc1YjlmNmViZWZhNjM2YThiYiA9ICQoJzxkaXYgaWQ9Imh0bWxfNzhiZDVlOWNiMzdlNGI3NWI5ZjZlYmVmYTYzNmE4YmIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgVG9yb250bywgTGF3cmVuY2UgUGFyazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2YyMTU2MzhlMWRmNGExMWE2NTQ3MDMyMGU4NjVjOTIuc2V0Q29udGVudChodG1sXzc4YmQ1ZTljYjM3ZTRiNzViOWY2ZWJlZmE2MzZhOGJiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzY4Y2I4YzIwZGFlZTRkZTE4ZWFhODQ5MTMyOTI3OWE1LmJpbmRQb3B1cChwb3B1cF8zZjIxNTYzOGUxZGY0YTExYTY1NDcwMzIwZTg2NWM5Mik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kOGVjYTYwMjM1NmU0N2FmOGRjMWRlYmIwODAxNzhlNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxMjc1MTEsLTc5LjM5MDE5NzVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfOGU0ZWUwOGMyZjVlNDRlMDliMmQxZmM5NzY2MzhhNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDEzMzVlNTAwNTJiNDNlYjg3ODcyMWRhMTJmZTRkYWUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNWViMGZmOTg2MzcxNDg2NTk3NmExMTZiNmQ0YjY0YjYgPSAkKCc8ZGl2IGlkPSJodG1sXzVlYjBmZjk4NjM3MTQ4NjU5NzZhMTE2YjZkNGI2NGI2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DZW50cmFsIFRvcm9udG8sIERhdmlzdmlsbGUgTm9ydGg8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQxMzM1ZTUwMDUyYjQzZWI4Nzg3MjFkYTEyZmU0ZGFlLnNldENvbnRlbnQoaHRtbF81ZWIwZmY5ODYzNzE0ODY1OTc2YTExNmI2ZDRiNjRiNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kOGVjYTYwMjM1NmU0N2FmOGRjMWRlYmIwODAxNzhlNi5iaW5kUG9wdXAocG9wdXBfNDEzMzVlNTAwNTJiNDNlYjg3ODcyMWRhMTJmZTRkYWUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjBhMGQ0YjVmOWU5NDkzN2I1OThlOTNjMjdmMjExODMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTUzODM0LC03OS40MDU2Nzg0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84ZTRlZTA4YzJmNWU0NGUwOWIyZDFmYzk3NjYzOGE0OCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hMTQxMTVmZjdlNjE0MTc3ODg1MTFlODAyYTE2Mzg0NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84OTRjZDBiYmUwOWU0NmY1YWNlOWQ2NDViNmY2MjkxYyA9ICQoJzxkaXYgaWQ9Imh0bWxfODk0Y2QwYmJlMDllNDZmNWFjZTlkNjQ1YjZmNjI5MWMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgVG9yb250bywgTm9ydGggVG9yb250byBXZXN0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hMTQxMTVmZjdlNjE0MTc3ODg1MTFlODAyYTE2Mzg0Ny5zZXRDb250ZW50KGh0bWxfODk0Y2QwYmJlMDllNDZmNWFjZTlkNjQ1YjZmNjI5MWMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjBhMGQ0YjVmOWU5NDkzN2I1OThlOTNjMjdmMjExODMuYmluZFBvcHVwKHBvcHVwX2ExNDExNWZmN2U2MTQxNzc4ODUxMWU4MDJhMTYzODQ3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFiZjM2YTAzNjYyMTQ1Njg5NDA0NzEyOGI2NGIyNjI5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA0MzI0NCwtNzkuMzg4NzkwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84ZTRlZTA4YzJmNWU0NGUwOWIyZDFmYzk3NjYzOGE0OCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lMjRhNzRmNTdmOTk0NjQzYTUxNjk2YWIzZTEyZjBlNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zYTY2OTA4YTljMDQ0YjBmODdlMzViNTU5Y2EzZjhkZSA9ICQoJzxkaXYgaWQ9Imh0bWxfM2E2NjkwOGE5YzA0NGIwZjg3ZTM1YjU1OWNhM2Y4ZGUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgVG9yb250bywgRGF2aXN2aWxsZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTI0YTc0ZjU3Zjk5NDY0M2E1MTY5NmFiM2UxMmYwZTYuc2V0Q29udGVudChodG1sXzNhNjY5MDhhOWMwNDRiMGY4N2UzNWI1NTljYTNmOGRlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFiZjM2YTAzNjYyMTQ1Njg5NDA0NzEyOGI2NGIyNjI5LmJpbmRQb3B1cChwb3B1cF9lMjRhNzRmNTdmOTk0NjQzYTUxNjk2YWIzZTEyZjBlNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yM2VhNTBiZDhkNmQ0OGVhODE3MzA5ODg0OWY3YzQ5ZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4OTU3NDMsLTc5LjM4MzE1OTkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzhlNGVlMDhjMmY1ZTQ0ZTA5YjJkMWZjOTc2NjM4YTQ4KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzM2ZTVkZmI3ZGViMDQxYjRhNDRiYzJhZmM3MTRkM2U1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhkNjkwNzQzODU5YjQxNTVhZjFhODBmNjZiMmRiM2ExID0gJCgnPGRpdiBpZD0iaHRtbF84ZDY5MDc0Mzg1OWI0MTU1YWYxYTgwZjY2YjJkYjNhMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBUb3JvbnRvLCBNb29yZSBQYXJrLCBTdW1tZXJoaWxsIEVhc3Q8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM2ZTVkZmI3ZGViMDQxYjRhNDRiYzJhZmM3MTRkM2U1LnNldENvbnRlbnQoaHRtbF84ZDY5MDc0Mzg1OWI0MTU1YWYxYTgwZjY2YjJkYjNhMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yM2VhNTBiZDhkNmQ0OGVhODE3MzA5ODg0OWY3YzQ5Zi5iaW5kUG9wdXAocG9wdXBfMzZlNWRmYjdkZWIwNDFiNGE0NGJjMmFmYzcxNGQzZTUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODkyNDVhZjI5NTRjNDdiM2FhYzIxOWIzN2ZmMWVjZjQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODY0MTIyOTk5OTk5OSwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84ZTRlZTA4YzJmNWU0NGUwOWIyZDFmYzk3NjYzOGE0OCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zMjM0ODVkMjg1YWI0MGU2OTU2YTIwZTZiNTlhNzQzOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zOTY0ZGQ4NDA5OWI0MWMyOTcyYzgzZmVjYTU5YmUxZiA9ICQoJzxkaXYgaWQ9Imh0bWxfMzk2NGRkODQwOTliNDFjMjk3MmM4M2ZlY2E1OWJlMWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgVG9yb250bywgRGVlciBQYXJrLCBGb3Jlc3QgSGlsbCBTRSwgUmF0aG5lbGx5LCBTb3V0aCBIaWxsLCBTdW1tZXJoaWxsIFdlc3Q8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzMyMzQ4NWQyODVhYjQwZTY5NTZhMjBlNmI1OWE3NDM4LnNldENvbnRlbnQoaHRtbF8zOTY0ZGQ4NDA5OWI0MWMyOTcyYzgzZmVjYTU5YmUxZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84OTI0NWFmMjk1NGM0N2IzYWFjMjE5YjM3ZmYxZWNmNC5iaW5kUG9wdXAocG9wdXBfMzIzNDg1ZDI4NWFiNDBlNjk1NmEyMGU2YjU5YTc0MzgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmI5MGMwZDY3ZjVhNGEwYTkyNjc2NjZlMGZhNmY4YWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Nzk1NjI2LC03OS4zNzc1Mjk0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84ZTRlZTA4YzJmNWU0NGUwOWIyZDFmYzk3NjYzOGE0OCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84NmU5NzZhMWY1MTE0ZWM2YmJlMjViM2Y1YjQxYjMyOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84YzA2ZGM5YWFiNDI0MzliYjE5MGMwYWIyNGQwYzRkMSA9ICQoJzxkaXYgaWQ9Imh0bWxfOGMwNmRjOWFhYjQyNDM5YmIxOTBjMGFiMjRkMGM0ZDEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG8sIFJvc2VkYWxlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84NmU5NzZhMWY1MTE0ZWM2YmJlMjViM2Y1YjQxYjMyOC5zZXRDb250ZW50KGh0bWxfOGMwNmRjOWFhYjQyNDM5YmIxOTBjMGFiMjRkMGM0ZDEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmI5MGMwZDY3ZjVhNGEwYTkyNjc2NjZlMGZhNmY4YWEuYmluZFBvcHVwKHBvcHVwXzg2ZTk3NmExZjUxMTRlYzZiYmUyNWIzZjViNDFiMzI4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc2N2ViZGI4ZWEyNjQ5YWQ5OTM2ZjcxNzY1MTA5ZTdkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY3OTY3LC03OS4zNjc2NzUzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzhlNGVlMDhjMmY1ZTQ0ZTA5YjJkMWZjOTc2NjM4YTQ4KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzk5OGViNjhlMDFmODRkMWY4MjVlZDFlYzk5NGRhZDQ3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzNjY2QxYTQ1ZWY1YzRiYzM5OTY5MjU5NDhiZWU5MWJjID0gJCgnPGRpdiBpZD0iaHRtbF8zY2NkMWE0NWVmNWM0YmMzOTk2OTI1OTQ4YmVlOTFiYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250bywgQ2FiYmFnZXRvd24sIFN0LiBKYW1lcyBUb3duPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85OThlYjY4ZTAxZjg0ZDFmODI1ZWQxZWM5OTRkYWQ0Ny5zZXRDb250ZW50KGh0bWxfM2NjZDFhNDVlZjVjNGJjMzk5NjkyNTk0OGJlZTkxYmMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzY3ZWJkYjhlYTI2NDlhZDk5MzZmNzE3NjUxMDllN2QuYmluZFBvcHVwKHBvcHVwXzk5OGViNjhlMDFmODRkMWY4MjVlZDFlYzk5NGRhZDQ3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBkYjUxMWU0M2I1NDRlZjZiMzBkZGIwNDE5NTZkZTM1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY1ODU5OSwtNzkuMzgzMTU5OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfOGU0ZWUwOGMyZjVlNDRlMDliMmQxZmM5NzY2MzhhNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWU4ODE2NWI3ZDdkNDM2MDhhZmRhNGE5NDA0NGMyYWQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZWI3ZmMwOGEzM2I3NDFlZjk5Y2FiM2U0OTRlODQwZTMgPSAkKCc8ZGl2IGlkPSJodG1sX2ViN2ZjMDhhMzNiNzQxZWY5OWNhYjNlNDk0ZTg0MGUzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvLCBDaHVyY2ggYW5kIFdlbGxlc2xleTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMWU4ODE2NWI3ZDdkNDM2MDhhZmRhNGE5NDA0NGMyYWQuc2V0Q29udGVudChodG1sX2ViN2ZjMDhhMzNiNzQxZWY5OWNhYjNlNDk0ZTg0MGUzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBkYjUxMWU0M2I1NDRlZjZiMzBkZGIwNDE5NTZkZTM1LmJpbmRQb3B1cChwb3B1cF8xZTg4MTY1YjdkN2Q0MzYwOGFmZGE0YTk0MDQ0YzJhZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81YjE4ZGM1ZjU4OWQ0ODZhYmZjNmM0N2VjYzcxN2E5YiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1NDI1OTksLTc5LjM2MDYzNTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfOGU0ZWUwOGMyZjVlNDRlMDliMmQxZmM5NzY2MzhhNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzQ3MmJjZGUwNjU5NDEzZThjODRlY2M0NDU2ZjlhZjIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWY1ZDBjMzY2MTc0NDAxZGExNDI1NmU0MmE3ZThiYjMgPSAkKCc8ZGl2IGlkPSJodG1sXzFmNWQwYzM2NjE3NDQwMWRhMTQyNTZlNDJhN2U4YmIzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvLCBIYXJib3VyZnJvbnQsIFJlZ2VudCBQYXJrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83NDcyYmNkZTA2NTk0MTNlOGM4NGVjYzQ0NTZmOWFmMi5zZXRDb250ZW50KGh0bWxfMWY1ZDBjMzY2MTc0NDAxZGExNDI1NmU0MmE3ZThiYjMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNWIxOGRjNWY1ODlkNDg2YWJmYzZjNDdlY2M3MTdhOWIuYmluZFBvcHVwKHBvcHVwXzc0NzJiY2RlMDY1OTQxM2U4Yzg0ZWNjNDQ1NmY5YWYyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzEzZDlkNmE3Mjc0MTQyY2NhMTFhZGU3NThiOGY4YmRlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3MTYxOCwtNzkuMzc4OTM3MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfOGU0ZWUwOGMyZjVlNDRlMDliMmQxZmM5NzY2MzhhNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTc4NGVjN2QyYzViNGQ4ZGIxZjEwNGM0YWVmNzc2NGYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjNhMjBmYTE3OWNkNGQxNWI0MzA4MzIxZDllMzcyYWEgPSAkKCc8ZGl2IGlkPSJodG1sX2IzYTIwZmExNzljZDRkMTViNDMwODMyMWQ5ZTM3MmFhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvLCBSeWVyc29uLCBHYXJkZW4gRGlzdHJpY3Q8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U3ODRlYzdkMmM1YjRkOGRiMWYxMDRjNGFlZjc3NjRmLnNldENvbnRlbnQoaHRtbF9iM2EyMGZhMTc5Y2Q0ZDE1YjQzMDgzMjFkOWUzNzJhYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xM2Q5ZDZhNzI3NDE0MmNjYTExYWRlNzU4YjhmOGJkZS5iaW5kUG9wdXAocG9wdXBfZTc4NGVjN2QyYzViNGQ4ZGIxZjEwNGM0YWVmNzc2NGYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTJjNGQwM2ZlNWQ0NDVjODk2MWVkYjMzODIzYjNhNjggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTE0OTM5LC03OS4zNzU0MTc5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzhlNGVlMDhjMmY1ZTQ0ZTA5YjJkMWZjOTc2NjM4YTQ4KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzUxY2JkNmMzNGFiODQxZDM5YTUzZTVlYzY0N2IzMTkwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ZhZDY1YWVlMmQ1MjQzN2JhNTlhYTE1MWViYjEwMDVmID0gJCgnPGRpdiBpZD0iaHRtbF9mYWQ2NWFlZTJkNTI0MzdiYTU5YWExNTFlYmIxMDA1ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250bywgU3QuIEphbWVzIFRvd248L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzUxY2JkNmMzNGFiODQxZDM5YTUzZTVlYzY0N2IzMTkwLnNldENvbnRlbnQoaHRtbF9mYWQ2NWFlZTJkNTI0MzdiYTU5YWExNTFlYmIxMDA1Zik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hMmM0ZDAzZmU1ZDQ0NWM4OTYxZWRiMzM4MjNiM2E2OC5iaW5kUG9wdXAocG9wdXBfNTFjYmQ2YzM0YWI4NDFkMzlhNTNlNWVjNjQ3YjMxOTApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOThkOGExZDMxMDkwNDRlYWFkMTAxM2I2YzMyNTBiMDMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDQ3NzA3OTk5OTk5OTYsLTc5LjM3MzMwNjRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfOGU0ZWUwOGMyZjVlNDRlMDliMmQxZmM5NzY2MzhhNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDA3MDhlNjVlMzE4NDZkNGE0ZDcxNWI1ZjcwOWJjMDcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTA2YjhiOGNiY2ZlNDM0M2FiOTljNDFhYmY4MGUyODYgPSAkKCc8ZGl2IGlkPSJodG1sXzkwNmI4YjhjYmNmZTQzNDNhYjk5YzQxYWJmODBlMjg2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvLCBCZXJjenkgUGFyazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDA3MDhlNjVlMzE4NDZkNGE0ZDcxNWI1ZjcwOWJjMDcuc2V0Q29udGVudChodG1sXzkwNmI4YjhjYmNmZTQzNDNhYjk5YzQxYWJmODBlMjg2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzk4ZDhhMWQzMTA5MDQ0ZWFhZDEwMTNiNmMzMjUwYjAzLmJpbmRQb3B1cChwb3B1cF80MDcwOGU2NWUzMTg0NmQ0YTRkNzE1YjVmNzA5YmMwNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lYzEyOTE5ZTVkZjI0NjZmOWI2YWM4YmUxMGZmNDRhNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1Nzk1MjQsLTc5LjM4NzM4MjZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfOGU0ZWUwOGMyZjVlNDRlMDliMmQxZmM5NzY2MzhhNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTllZmI1NmU0Mjc0NDBiMDk2ZTM3Yjc2MTc4ODc2OTMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZmQxZGUzYjdlZDM1NGIzNTliM2E0MzlmZTgxYzI0MTQgPSAkKCc8ZGl2IGlkPSJodG1sX2ZkMWRlM2I3ZWQzNTRiMzU5YjNhNDM5ZmU4MWMyNDE0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvLCBDZW50cmFsIEJheSBTdHJlZXQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U5ZWZiNTZlNDI3NDQwYjA5NmUzN2I3NjE3ODg3NjkzLnNldENvbnRlbnQoaHRtbF9mZDFkZTNiN2VkMzU0YjM1OWIzYTQzOWZlODFjMjQxNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lYzEyOTE5ZTVkZjI0NjZmOWI2YWM4YmUxMGZmNDRhNy5iaW5kUG9wdXAocG9wdXBfZTllZmI1NmU0Mjc0NDBiMDk2ZTM3Yjc2MTc4ODc2OTMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTkzODUwOTYxMzNhNDVhMDg4NjMwOTNlZTkwYTMwNzkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTA1NzEyMDAwMDAwMSwtNzkuMzg0NTY3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84ZTRlZTA4YzJmNWU0NGUwOWIyZDFmYzk3NjYzOGE0OCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mMzJiMjdiYTRiMDk0ZDBkOGRhNDZmNDVjZmU2OTM3YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85MDdlNjBiNGM3YTQ0ODU4YTVjYzYxZWJmMGNkMTY4ZSA9ICQoJzxkaXYgaWQ9Imh0bWxfOTA3ZTYwYjRjN2E0NDg1OGE1Y2M2MWViZjBjZDE2OGUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG8sIEFkZWxhaWRlLCBLaW5nLCBSaWNobW9uZDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjMyYjI3YmE0YjA5NGQwZDhkYTQ2ZjQ1Y2ZlNjkzN2Euc2V0Q29udGVudChodG1sXzkwN2U2MGI0YzdhNDQ4NThhNWNjNjFlYmYwY2QxNjhlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzU5Mzg1MDk2MTMzYTQ1YTA4ODYzMDkzZWU5MGEzMDc5LmJpbmRQb3B1cChwb3B1cF9mMzJiMjdiYTRiMDk0ZDBkOGRhNDZmNDVjZmU2OTM3YSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xNTAyMjBiNjhkNDU0OWM2OWM2YzM0NzAxZGYxMzI3MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0MDgxNTcsLTc5LjM4MTc1MjI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzhlNGVlMDhjMmY1ZTQ0ZTA5YjJkMWZjOTc2NjM4YTQ4KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RmYmE3MTU3NDI0OTRmNWI5NTkzMTA2NjczYzY2MTliID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzg2ODczMWRlODQ5NzRlMjA5M2E0ODFjZjI1YjU0YTExID0gJCgnPGRpdiBpZD0iaHRtbF84Njg3MzFkZTg0OTc0ZTIwOTNhNDgxY2YyNWI1NGExMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250bywgSGFyYm91cmZyb250IEVhc3QsIFRvcm9udG8gSXNsYW5kcywgVW5pb24gU3RhdGlvbjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGZiYTcxNTc0MjQ5NGY1Yjk1OTMxMDY2NzNjNjYxOWIuc2V0Q29udGVudChodG1sXzg2ODczMWRlODQ5NzRlMjA5M2E0ODFjZjI1YjU0YTExKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzE1MDIyMGI2OGQ0NTQ5YzY5YzZjMzQ3MDFkZjEzMjcxLmJpbmRQb3B1cChwb3B1cF9kZmJhNzE1NzQyNDk0ZjViOTU5MzEwNjY3M2M2NjE5Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kNzIyNjJlMDBiMGE0MWU5YTllZTc1MmFlN2Y3NmU4NyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzE3NjgsLTc5LjM4MTU3NjQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzhlNGVlMDhjMmY1ZTQ0ZTA5YjJkMWZjOTc2NjM4YTQ4KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2YzNzYzZjRhMDY4MTRiNDg5MTZiODBhMzRhOTc2OGM3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRhYjE5MTk1M2VhNzRkNDk4YzY1NGI3Zjk0MWI5MWExID0gJCgnPGRpdiBpZD0iaHRtbF80YWIxOTE5NTNlYTc0ZDQ5OGM2NTRiN2Y5NDFiOTFhMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250bywgRGVzaWduIEV4Y2hhbmdlLCBUb3JvbnRvIERvbWluaW9uIENlbnRyZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjM3NjNmNGEwNjgxNGI0ODkxNmI4MGEzNGE5NzY4Yzcuc2V0Q29udGVudChodG1sXzRhYjE5MTk1M2VhNzRkNDk4YzY1NGI3Zjk0MWI5MWExKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Q3MjI2MmUwMGIwYTQxZTlhOWVlNzUyYWU3Zjc2ZTg3LmJpbmRQb3B1cChwb3B1cF9mMzc2M2Y0YTA2ODE0YjQ4OTE2YjgwYTM0YTk3NjhjNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wNWY3Y2NjZTI2MmI0NTM5ODcxZDJkY2ZjN2M3NzEzZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0ODE5ODUsLTc5LjM3OTgxNjkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzhlNGVlMDhjMmY1ZTQ0ZTA5YjJkMWZjOTc2NjM4YTQ4KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzI3MGEwYWJiMDZkNTRmMjY5MWUwYjAyODY0NzkwMGE3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FiZjk3N2NhZTk5OTQ4MmQ5N2JiZmNiMTNjMDE1NGJmID0gJCgnPGRpdiBpZD0iaHRtbF9hYmY5NzdjYWU5OTk0ODJkOTdiYmZjYjEzYzAxNTRiZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250bywgQ29tbWVyY2UgQ291cnQsIFZpY3RvcmlhIEhvdGVsPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yNzBhMGFiYjA2ZDU0ZjI2OTFlMGIwMjg2NDc5MDBhNy5zZXRDb250ZW50KGh0bWxfYWJmOTc3Y2FlOTk5NDgyZDk3YmJmY2IxM2MwMTU0YmYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDVmN2NjY2UyNjJiNDUzOTg3MWQyZGNmYzdjNzcxM2YuYmluZFBvcHVwKHBvcHVwXzI3MGEwYWJiMDZkNTRmMjY5MWUwYjAyODY0NzkwMGE3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzE2YzEyYWVjYTNiNzRmODhiYjA4N2JiZGFmMDZmNWUzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzExNjk0OCwtNzkuNDE2OTM1NTk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfOGU0ZWUwOGMyZjVlNDRlMDliMmQxZmM5NzY2MzhhNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzE4MDQyNmViZTNmNGRjYmI1MGJhMGFlNDljY2VmOWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWEyNTE1MGRkYjQyNGY2ZTk2MjAwY2NlODRjNDE0MWYgPSAkKCc8ZGl2IGlkPSJodG1sXzFhMjUxNTBkZGI0MjRmNmU5NjIwMGNjZTg0YzQxNDFmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DZW50cmFsIFRvcm9udG8sIFJvc2VsYXduPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jMTgwNDI2ZWJlM2Y0ZGNiYjUwYmEwYWU0OWNjZWY5Zi5zZXRDb250ZW50KGh0bWxfMWEyNTE1MGRkYjQyNGY2ZTk2MjAwY2NlODRjNDE0MWYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTZjMTJhZWNhM2I3NGY4OGJiMDg3YmJkYWYwNmY1ZTMuYmluZFBvcHVwKHBvcHVwX2MxODA0MjZlYmUzZjRkY2JiNTBiYTBhZTQ5Y2NlZjlmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Y0OGU1Yjk5YTc5ZjRlMzFiOWQzOTMzNTQxOTBlOGI3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjk2OTQ3NiwtNzkuNDExMzA3MjAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfOGU0ZWUwOGMyZjVlNDRlMDliMmQxZmM5NzY2MzhhNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTM2Nzk1MmIwYzUzNGVhMjg5OGEwODYyYTFkNDJhMzUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzQ3MjhkNjQ1ZDYxNGIwYThhODdjYjZhZDJkZGFiODggPSAkKCc8ZGl2IGlkPSJodG1sXzc0NzI4ZDY0NWQ2MTRiMGE4YTg3Y2I2YWQyZGRhYjg4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DZW50cmFsIFRvcm9udG8sIEZvcmVzdCBIaWxsIE5vcnRoLCBGb3Jlc3QgSGlsbCBXZXN0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xMzY3OTUyYjBjNTM0ZWEyODk4YTA4NjJhMWQ0MmEzNS5zZXRDb250ZW50KGh0bWxfNzQ3MjhkNjQ1ZDYxNGIwYThhODdjYjZhZDJkZGFiODgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjQ4ZTViOTlhNzlmNGUzMWI5ZDM5MzM1NDE5MGU4YjcuYmluZFBvcHVwKHBvcHVwXzEzNjc5NTJiMGM1MzRlYTI4OThhMDg2MmExZDQyYTM1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzA5NWIwYzkyZTgzMzRjYmVhMGIxOTA3NTVhNDA2Y2Q5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjcyNzA5NywtNzkuNDA1Njc4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfOGU0ZWUwOGMyZjVlNDRlMDliMmQxZmM5NzY2MzhhNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTM4OGJjYTdlOGI3NDZmMjk2MzdkM2MzYTllN2Y0ODUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYmE3NGU4ZGRjMjdjNGM3Y2JhYTYzOTAzZmQ3MTMxZTUgPSAkKCc8ZGl2IGlkPSJodG1sX2JhNzRlOGRkYzI3YzRjN2NiYWE2MzkwM2ZkNzEzMWU1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DZW50cmFsIFRvcm9udG8sIFRoZSBBbm5leCwgTm9ydGggTWlkdG93biwgWW9ya3ZpbGxlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xMzg4YmNhN2U4Yjc0NmYyOTYzN2QzYzNhOWU3ZjQ4NS5zZXRDb250ZW50KGh0bWxfYmE3NGU4ZGRjMjdjNGM3Y2JhYTYzOTAzZmQ3MTMxZTUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDk1YjBjOTJlODMzNGNiZWEwYjE5MDc1NWE0MDZjZDkuYmluZFBvcHVwKHBvcHVwXzEzODhiY2E3ZThiNzQ2ZjI5NjM3ZDNjM2E5ZTdmNDg1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzE3MjQ1ZjNlNmFiZTRlOWE5ZWI3Mjc3ZDU0MGI0YzEyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyNjk1NiwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84ZTRlZTA4YzJmNWU0NGUwOWIyZDFmYzk3NjYzOGE0OCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xNzBjMDA4NjZkNTU0MzFjYjZlODgwODA0NTZkNTA3ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jMGNiNDRmODdiNzc0MzAzYTJlYjdjNzRmOTk3ZjA0NiA9ICQoJzxkaXYgaWQ9Imh0bWxfYzBjYjQ0Zjg3Yjc3NDMwM2EyZWI3Yzc0Zjk5N2YwNDYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG8sIEhhcmJvcmQsIFVuaXZlcnNpdHkgb2YgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTcwYzAwODY2ZDU1NDMxY2I2ZTg4MDgwNDU2ZDUwN2Yuc2V0Q29udGVudChodG1sX2MwY2I0NGY4N2I3NzQzMDNhMmViN2M3NGY5OTdmMDQ2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzE3MjQ1ZjNlNmFiZTRlOWE5ZWI3Mjc3ZDU0MGI0YzEyLmJpbmRQb3B1cChwb3B1cF8xNzBjMDA4NjZkNTU0MzFjYjZlODgwODA0NTZkNTA3Zik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83ZDA0MDhiMzcxMjM0NWQ1YWE2ZmY4ZTNkMGM2Y2NlZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MzIwNTcsLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfOGU0ZWUwOGMyZjVlNDRlMDliMmQxZmM5NzY2MzhhNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfY2FiNTM0ZGJkYTlmNDAxMWEzYzE4YTE4YWI1Y2Q3N2UgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTc2NzdlZWY2MDgyNDhkNDgxYWU0ZjFkNDQ1Mzg1MjggPSAkKCc8ZGl2IGlkPSJodG1sXzE3Njc3ZWVmNjA4MjQ4ZDQ4MWFlNGYxZDQ0NTM4NTI4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvLCBDaGluYXRvd24sIEdyYW5nZSBQYXJrLCBLZW5zaW5ndG9uIE1hcmtldDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2FiNTM0ZGJkYTlmNDAxMWEzYzE4YTE4YWI1Y2Q3N2Uuc2V0Q29udGVudChodG1sXzE3Njc3ZWVmNjA4MjQ4ZDQ4MWFlNGYxZDQ0NTM4NTI4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdkMDQwOGIzNzEyMzQ1ZDVhYTZmZjhlM2QwYzZjY2VmLmJpbmRQb3B1cChwb3B1cF9jYWI1MzRkYmRhOWY0MDExYTNjMThhMThhYjVjZDc3ZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80YjIzNTZlY2Y1YTA0NzQ0ODI1YWEzYjk1YTFlNzRiZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODk0NjcsLTc5LjM5NDQxOTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfOGU0ZWUwOGMyZjVlNDRlMDliMmQxZmM5NzY2MzhhNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjc3MGU5ZjE0YzQ0NDhmNzkxMTIxMzQ0MjZkNTRjMDkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjQ3Y2M5YTIzMTU1NGY5Y2I4ZGIzOTNiM2RlNTY1NWYgPSAkKCc8ZGl2IGlkPSJodG1sX2Y0N2NjOWEyMzE1NTRmOWNiOGRiMzkzYjNkZTU2NTVmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvLCBDTiBUb3dlciwgQmF0aHVyc3QgUXVheSwgSXNsYW5kIGFpcnBvcnQsIEhhcmJvdXJmcm9udCBXZXN0LCBLaW5nIGFuZCBTcGFkaW5hLCBSYWlsd2F5IExhbmRzLCBTb3V0aCBOaWFnYXJhPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iNzcwZTlmMTRjNDQ0OGY3OTExMjEzNDQyNmQ1NGMwOS5zZXRDb250ZW50KGh0bWxfZjQ3Y2M5YTIzMTU1NGY5Y2I4ZGIzOTNiM2RlNTY1NWYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNGIyMzU2ZWNmNWEwNDc0NDgyNWFhM2I5NWExZTc0YmYuYmluZFBvcHVwKHBvcHVwX2I3NzBlOWYxNGM0NDQ4Zjc5MTEyMTM0NDI2ZDU0YzA5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBjOGJhNjFiYTM0NjQzMWE4ZjdkYmNkMjE3MTMyMGM3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ2NDM1MiwtNzkuMzc0ODQ1OTk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfOGU0ZWUwOGMyZjVlNDRlMDliMmQxZmM5NzY2MzhhNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDVmMzQzNzllNGMyNGY4ODljM2JlMDA4YzMxOTAwODYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTg3ZTkxMmQ1YjZiNDU3YmE3ZTY2ODc1YzE1YmQ1ZTMgPSAkKCc8ZGl2IGlkPSJodG1sXzk4N2U5MTJkNWI2YjQ1N2JhN2U2Njg3NWMxNWJkNWUzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvLCBTdG4gQSBQTyBCb3hlcyAyNSBUaGUgRXNwbGFuYWRlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kNWYzNDM3OWU0YzI0Zjg4OWMzYmUwMDhjMzE5MDA4Ni5zZXRDb250ZW50KGh0bWxfOTg3ZTkxMmQ1YjZiNDU3YmE3ZTY2ODc1YzE1YmQ1ZTMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMGM4YmE2MWJhMzQ2NDMxYThmN2RiY2QyMTcxMzIwYzcuYmluZFBvcHVwKHBvcHVwX2Q1ZjM0Mzc5ZTRjMjRmODg5YzNiZTAwOGMzMTkwMDg2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ZhMTJlYjdjYTQ1ZjQ2OWJiZDAyOWM2Mzc5Njc2N2UwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4NDI5MiwtNzkuMzgyMjgwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84ZTRlZTA4YzJmNWU0NGUwOWIyZDFmYzk3NjYzOGE0OCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xNjFhMDU0MWY5ZTg0MGYxYjc4MWZjYmJiMmY2MDYyOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wYmYzNzk5ZTRjYTk0ODQyOWU0MmRmMTRlODI3OTIzOSA9ICQoJzxkaXYgaWQ9Imh0bWxfMGJmMzc5OWU0Y2E5NDg0MjllNDJkZjE0ZTgyNzkyMzkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG8sIEZpcnN0IENhbmFkaWFuIFBsYWNlLCBVbmRlcmdyb3VuZCBjaXR5PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xNjFhMDU0MWY5ZTg0MGYxYjc4MWZjYmJiMmY2MDYyOS5zZXRDb250ZW50KGh0bWxfMGJmMzc5OWU0Y2E5NDg0MjllNDJkZjE0ZTgyNzkyMzkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmExMmViN2NhNDVmNDY5YmJkMDI5YzYzNzk2NzY3ZTAuYmluZFBvcHVwKHBvcHVwXzE2MWEwNTQxZjllODQwZjFiNzgxZmNiYmIyZjYwNjI5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc3YWYxYzk2NTE0NTQ2NTM5OWYwNTY3ZWQyYjNhNzVhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY5NTQyLC03OS40MjI1NjM3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzhlNGVlMDhjMmY1ZTQ0ZTA5YjJkMWZjOTc2NjM4YTQ4KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzg4YjZlZjQxNWViOTRlMWFiZDBmZDE5MDc2MWI3MDBiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2VmY2IxYTljNDhmYTQwODRhMDYyZDBiZTQzZjYzYjBmID0gJCgnPGRpdiBpZD0iaHRtbF9lZmNiMWE5YzQ4ZmE0MDg0YTA2MmQwYmU0M2Y2M2IwZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250bywgQ2hyaXN0aWU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg4YjZlZjQxNWViOTRlMWFiZDBmZDE5MDc2MWI3MDBiLnNldENvbnRlbnQoaHRtbF9lZmNiMWE5YzQ4ZmE0MDg0YTA2MmQwYmU0M2Y2M2IwZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83N2FmMWM5NjUxNDU0NjUzOTlmMDU2N2VkMmIzYTc1YS5iaW5kUG9wdXAocG9wdXBfODhiNmVmNDE1ZWI5NGUxYWJkMGZkMTkwNzYxYjcwMGIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTk1Y2Y5NDVhOTc3NDQ2YWFmM2I2YzI0M2Y4YmQ2ODEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjkwMDUxMDAwMDAwMSwtNzkuNDQyMjU5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84ZTRlZTA4YzJmNWU0NGUwOWIyZDFmYzk3NjYzOGE0OCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kZTQ3NWE1YWUzOWY0MDM2ODdlYTc1MjA3YTUxYzFmOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85ZmJhOWY0ZTg0ZDA0ZTA4YjhkNDAwNDViMjM3OTgxYSA9ICQoJzxkaXYgaWQ9Imh0bWxfOWZiYTlmNGU4NGQwNGUwOGI4ZDQwMDQ1YjIzNzk4MWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldlc3QgVG9yb250bywgRG92ZXJjb3VydCBWaWxsYWdlLCBEdWZmZXJpbjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGU0NzVhNWFlMzlmNDAzNjg3ZWE3NTIwN2E1MWMxZjguc2V0Q29udGVudChodG1sXzlmYmE5ZjRlODRkMDRlMDhiOGQ0MDA0NWIyMzc5ODFhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzE5NWNmOTQ1YTk3NzQ0NmFhZjNiNmMyNDNmOGJkNjgxLmJpbmRQb3B1cChwb3B1cF9kZTQ3NWE1YWUzOWY0MDM2ODdlYTc1MjA3YTUxYzFmOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83Zjg0MTRhYWRhNDU0ZjNjYWZmMjY5YTJmNmZkZjM5OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzkyNjcwMDAwMDAwNiwtNzkuNDE5NzQ5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84ZTRlZTA4YzJmNWU0NGUwOWIyZDFmYzk3NjYzOGE0OCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84OTJlNjU5ZmNkMGM0ZDY3ODkyZWI3YTQ5OGEzNDhjMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jOWRiMDg2ODYxMTg0NWNkOGM3MmQ5NDVhYmU4OWQyNCA9ICQoJzxkaXYgaWQ9Imh0bWxfYzlkYjA4Njg2MTE4NDVjZDhjNzJkOTQ1YWJlODlkMjQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldlc3QgVG9yb250bywgTGl0dGxlIFBvcnR1Z2FsLCBUcmluaXR5PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84OTJlNjU5ZmNkMGM0ZDY3ODkyZWI3YTQ5OGEzNDhjMC5zZXRDb250ZW50KGh0bWxfYzlkYjA4Njg2MTE4NDVjZDhjNzJkOTQ1YWJlODlkMjQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfN2Y4NDE0YWFkYTQ1NGYzY2FmZjI2OWEyZjZmZGYzOTkuYmluZFBvcHVwKHBvcHVwXzg5MmU2NTlmY2QwYzRkNjc4OTJlYjdhNDk4YTM0OGMwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Y1Y2NhOTIyMmM2NjRhNmM4MGQxYjBiNTQwNTBlZWY4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjM2ODQ3MiwtNzkuNDI4MTkxNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfOGU0ZWUwOGMyZjVlNDRlMDliMmQxZmM5NzY2MzhhNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZmUzYWE1NTJjNGQ2NDc5OTg1OGQwY2E0ZGZjY2E3MzEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNjAwN2YyZWUxZTBmNDEzNWI4MDA1YzNlYTY4MWE2MTAgPSAkKCc8ZGl2IGlkPSJodG1sXzYwMDdmMmVlMWUwZjQxMzViODAwNWMzZWE2ODFhNjEwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XZXN0IFRvcm9udG8sIEJyb2NrdG9uLCBFeGhpYml0aW9uIFBsYWNlLCBQYXJrZGFsZSBWaWxsYWdlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mZTNhYTU1MmM0ZDY0Nzk5ODU4ZDBjYTRkZmNjYTczMS5zZXRDb250ZW50KGh0bWxfNjAwN2YyZWUxZTBmNDEzNWI4MDA1YzNlYTY4MWE2MTApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjVjY2E5MjIyYzY2NGE2YzgwZDFiMGI1NDA1MGVlZjguYmluZFBvcHVwKHBvcHVwX2ZlM2FhNTUyYzRkNjQ3OTk4NThkMGNhNGRmY2NhNzMxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2JmZmI2YzI2YTI0MjQwM2RhMGQ1ZDVmOTAzMGY5OGU5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYxNjA4MywtNzkuNDY0NzYzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfOGU0ZWUwOGMyZjVlNDRlMDliMmQxZmM5NzY2MzhhNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMGI2OGFlNWZkOWU4NDkxODk4ZGE0ZjlkMzhhZmU2NTkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfY2UzYzM5ZGQwZTZmNGJmOGEzOWU0ZjA4YTVjY2M5MDAgPSAkKCc8ZGl2IGlkPSJodG1sX2NlM2MzOWRkMGU2ZjRiZjhhMzllNGYwOGE1Y2NjOTAwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XZXN0IFRvcm9udG8sIEhpZ2ggUGFyaywgVGhlIEp1bmN0aW9uIFNvdXRoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wYjY4YWU1ZmQ5ZTg0OTE4OThkYTRmOWQzOGFmZTY1OS5zZXRDb250ZW50KGh0bWxfY2UzYzM5ZGQwZTZmNGJmOGEzOWU0ZjA4YTVjY2M5MDApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYmZmYjZjMjZhMjQyNDAzZGEwZDVkNWY5MDMwZjk4ZTkuYmluZFBvcHVwKHBvcHVwXzBiNjhhZTVmZDllODQ5MTg5OGRhNGY5ZDM4YWZlNjU5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzAwOTU2ZmYwN2M1NTRkOTFiNzg0MTE5MDMyZjUyYzM1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4OTU5NywtNzkuNDU2MzI1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzhlNGVlMDhjMmY1ZTQ0ZTA5YjJkMWZjOTc2NjM4YTQ4KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzkxYjI1NWYwYWQyNzRiMzk5YjRmZmU0YTJhMDJhMjMyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzM0MjRkZDFhN2M2MTRkYWQ5YzcxZjU5ZDgyYjUxNTk1ID0gJCgnPGRpdiBpZD0iaHRtbF8zNDI0ZGQxYTdjNjE0ZGFkOWM3MWY1OWQ4MmI1MTU5NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2VzdCBUb3JvbnRvLCBQYXJrZGFsZSwgUm9uY2VzdmFsbGVzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85MWIyNTVmMGFkMjc0YjM5OWI0ZmZlNGEyYTAyYTIzMi5zZXRDb250ZW50KGh0bWxfMzQyNGRkMWE3YzYxNGRhZDljNzFmNTlkODJiNTE1OTUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDA5NTZmZjA3YzU1NGQ5MWI3ODQxMTkwMzJmNTJjMzUuYmluZFBvcHVwKHBvcHVwXzkxYjI1NWYwYWQyNzRiMzk5YjRmZmU0YTJhMDJhMjMyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzk2OTZlYWVmNjBkYjRjMzY5MzZkMGIyY2M1MTQ1ODZjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUxNTcwNiwtNzkuNDg0NDQ5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84ZTRlZTA4YzJmNWU0NGUwOWIyZDFmYzk3NjYzOGE0OCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hMWNmMDljMjNlN2Q0NjY0YTMwYjdkYzdhMzQ2MjY2NCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wZTI2YWFiZjQ0NjA0ZTc5ODBhNTg0ZjA5ZTc3Zjc1MSA9ICQoJzxkaXYgaWQ9Imh0bWxfMGUyNmFhYmY0NDYwNGU3OTgwYTU4NGYwOWU3N2Y3NTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldlc3QgVG9yb250bywgUnVubnltZWRlLCBTd2Fuc2VhPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hMWNmMDljMjNlN2Q0NjY0YTMwYjdkYzdhMzQ2MjY2NC5zZXRDb250ZW50KGh0bWxfMGUyNmFhYmY0NDYwNGU3OTgwYTU4NGYwOWU3N2Y3NTEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTY5NmVhZWY2MGRiNGMzNjkzNmQwYjJjYzUxNDU4NmMuYmluZFBvcHVwKHBvcHVwX2ExY2YwOWMyM2U3ZDQ2NjRhMzBiN2RjN2EzNDYyNjY0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2IyNGYwMGExYTZjMDRhNzhiNGZmMmRkOWUxNjNiMjk2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyNzQzOSwtNzkuMzIxNTU4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzhlNGVlMDhjMmY1ZTQ0ZTA5YjJkMWZjOTc2NjM4YTQ4KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2QwZGVhMzYxYWQ1MzQ1ZWE4ZTliYWMzY2RiNjM5Zjc0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RhN2M3OWU3MjUyYTRkMzQ4MzY1MDIwNzMwMjA4NWY1ID0gJCgnPGRpdiBpZD0iaHRtbF9kYTdjNzllNzI1MmE0ZDM0ODM2NTAyMDczMDIwODVmNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RWFzdCBUb3JvbnRvLCBCdXNpbmVzcyBSZXBseSBNYWlsIFByb2Nlc3NpbmcgQ2VudHJlIDk2OSBFYXN0ZXJuPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kMGRlYTM2MWFkNTM0NWVhOGU5YmFjM2NkYjYzOWY3NC5zZXRDb250ZW50KGh0bWxfZGE3Yzc5ZTcyNTJhNGQzNDgzNjUwMjA3MzAyMDg1ZjUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjI0ZjAwYTFhNmMwNGE3OGI0ZmYyZGQ5ZTE2M2IyOTYuYmluZFBvcHVwKHBvcHVwX2QwZGVhMzYxYWQ1MzQ1ZWE4ZTliYWMzY2RiNjM5Zjc0KTsKCiAgICAgICAgICAgIAogICAgICAgIAo8L3NjcmlwdD4=" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python

```


```python

```
