# after downloading the data, please unpack the data by running the following script
## you can remove names in the list to omit unpacking certain individuals's data.

# conan: https://drive.google.com/u/0/uc?id=1EqPuF28OdJ-pu3h8mG9QDOaYoWm4xf78
# devi2: https://drive.google.com/u/0/uc?id=1VPv5HUWz8s72XWWAlqIEyhoi8dpDDWEB
# fallon: https://drive.google.com/u/0/uc?id=1N8dtj72NJVN0fmzIxMraceK7FIT8aXx8
# kimmel: https://drive.google.com/u/0/uc?id=1FhUjaekGFmgzTu_7F89JT5_BAEw9ewSQ 
# stephen: https://drive.google.com/u/0/uc?id=1dMW1_aUAbCH_v6qS6alULebr7_IpQm-w
# trevor: https://drive.google.com/u/0/uc?id=1I2DhvZDteaOPGcJOeh_0gUkPRhV68ChW

NAMES=("conan" "devi2" "fallon" "kimmel" "stephen" "trevor")
for NAME in "${NAMES[@]}"; do
    if [ -f data/${NAME}_data.tar ] then
    	tar xvf data/${NAME}_data.tar -C data/ && rm data/${NAME}_data.tar 
    fi
done
