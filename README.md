The growing sport of disc golf is a variation on traditional (ball) golf, but with players attempting to throw flying discs at a target instead of attempting to hit a ball into a hole. The Professional Disc Golf Assocation (PDGA) approves discs for professional play and maintains a registry of a few physical characteristics of registered discs. The image below, taken from the PDGA technical standards document, shows the cross section of a disc with a number of measurements marked, several of which are registered with the PDGA.

![](measurements.png)

The variation in disc shapes leads to different flight characteristics. Some discs fly faster than others, some have more glide, etc. Many disc manufacturers do their best to summmarize these flight characteristics with four "flight numbers"—speed, glide, turn, and fade. This short project aims to use polynomial regression on the PDGA-registered measurements to predict the flight numbers of Innova-brand discs.

# The data

The PDGA makes their database of physical characteristics available for download, which we named `pdga.csv`. The flight numbers assigned to identical discs vary from manufacturer to manufacturer, so we chose to only analyze discs from one manufactuerer. Innova has the most PDGA-approved discs among all manufacturers, and many in the sport look to model their flight numbers after Innova's, so they were the natural choice to analyze. 

Innova has 157 PDGA-approved discs, which we can import from `pdga.csv`:
```python
>>> pdga = get_df_by_mfr('Innova Champion Discs').sort_index()
>>> pdga
            max_weight  diameter  height  ...  rim_depth_to_diameter  rim_config  flexibility
model                                     ...                                                
Ace              175.1      21.1     1.6  ...                    5.7       27.75        10.09
Aero             180.1      21.7     2.5  ...                    6.0       43.75         8.16
Animal           176.0      21.2     2.1  ...                    6.1       64.00         9.66
Ape              175.1      21.1     1.5  ...                    5.7       27.75        12.13
Apple            200.0      26.4     3.2  ...                    7.6       93.25         4.99
...                ...       ...     ...  ...                    ...         ...          ...
Wombat3          180.9      21.8     2.0  ...                    6.4       30.00         7.26
Wraith           175.1      21.1     1.4  ...                    5.7       26.75         8.85
XCaliber         175.1      21.1     1.6  ...                    5.7       29.50         8.85
XT V-Aviar       175.1      21.1     2.0  ...                    6.6       55.75         8.85
Zephyr           200.0      24.1     2.7  ...                    7.1       83.00        10.00

[157 rows x 9 columns]
```
The dataframe is indexed by the name of the disc, and the columns correspond to the 9 physical features registered by the PDGA.
```python
>>> for col in sorted(pdga.columns):
...     print(col)
... 
diameter
flexibility
height
inside_rim_diameter
max_weight
rim_config
rim_depth
rim_depth_to_diameter
rim_thickness
```

The flight numbers of Innova discs weren't as easily available as the physical characteristics. We obtained many (106) of them off the [Innova website](https://www.innovadiscs.com/disc-golf-discs/disc-comparison/) and were able to find 21 more through internet searches. These flight numbers are maintained in `innova.csv`, and a dataframe with both the PDGA-registered features and the flight numbers is made available with `get_innova_df()`:
```python
>>> df = get_innova_df()
>>> df
           max_weight  diameter  height  rim_depth  ...  speed  glide  turn  fade
model                                               ...                          
Ace             175.1      21.1     1.6        1.2  ...    2.0    3.0  -2.0   1.0
Aero            180.1      21.7     2.5        1.3  ...    3.0    6.0   0.0   0.0
Animal          176.0      21.2     2.1        1.3  ...    2.0    1.0   0.0   1.0
Ape             175.1      21.1     1.5        1.2  ...   13.0    5.0   0.0   4.0
Archangel       175.1      21.1     1.6        1.5  ...    8.0    6.0  -4.0   1.0
...               ...       ...     ...        ...  ...    ...    ...   ...   ...
Wombat          180.9      21.8     2.2        1.4  ...    5.0    6.0  -1.0   0.0
Wombat3         180.9      21.8     2.0        1.4  ...    5.0    6.0  -1.0   0.0
Wraith          175.1      21.1     1.4        1.2  ...   11.0    5.0  -1.0   3.0
XCaliber        175.1      21.1     1.6        1.2  ...   12.0    5.0   0.0   4.0
Zephyr          200.0      24.1     2.7        1.7  ...    2.0    3.0   0.0   0.0

[112 rows x 13 columns]
```

# Linear Regression
## Speed

The "speed" of a disc is a measure of how quickly the disc moves through the air. The speed of a disc is relatively well-predicted by the width of the rim of the disc.

![](Figures/1d/degree-1-speed-rim_thickness.png)
![](Figures/1d/degree-2-speed-rim_thickness.png)

## Glide

The "glide" of a disc is a measure of how much lift the disc generates and thus how long it will stay airborne. The glide of a disc is not well-explained by any of

![](Figures/1d/degree-1-speed-rim_thickness.png)
![](Figures/1d/degree-2-speed-rim_thickness.png)

