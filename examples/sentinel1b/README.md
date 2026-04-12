# Sentinel-1B Power Regulator Anomaly

This case study models the December 2021 Sentinel-1B satellite mission failure. 

## The Anomaly
On December 23, 2021, the Sentinel-1B Synthetic Aperture Radar (SAR) instrument experienced a sudden failure, preventing further operation. ESA investigations alongside the Anomaly Review Board concluded that the most likely root cause was a failure in the C-SAR Antenna Power Supply (CAPS) unit—specifically the regulated 28V bus. 

## The Model
In Aethelix, we model this anomaly as an unexpected catastrophic dropout in the bus voltage. 
A dynamically inserted `caps_regulator_failure` root cause connects directly to standard intermediate nodes (`bus_regulation`) and (`payload_temp`) which reflects what happens off-pipeline.
This demonstrates how Aethelix can be efficiently extended with new, custom-tailored nodes specific to historical case studies or specific mission profiles.
