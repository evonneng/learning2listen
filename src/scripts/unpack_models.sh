## models can be downloaded at:
# conan: https://drive.google.com/u/0/uc?id=1YNvDut219jjXTYpeGUo7ka8boHUufKqa
# fallon: https://drive.google.com/u/0/uc?id=14ADfugN54rRG6gYf7trw91-52Pk_pGsP
# stephen: https://drive.google.com/u/0/uc?id=1KAGdXb8e8gO7387t9zlYH8I6rYS_bOdd
# trevor: https://drive.google.com/u/0/uc?id=1nIAWjlydSxNjtJFevzwtq4LS8TquKeg2

# conan models:
## config files are already provided under: configs/vq/ and vqgan/configs
## associated conan models can be properly untarred using the following script:
tar xvf models/conan_models.tar && rm models/conan_models.tar

## remaining models and config files for other speakers
NAMES=("fallon" "stephen" "trevor")
for NAME in "${NAMES[@]}"; do
    tar xvf models/${NAME}_models.tar && rm models/${NAME}_models.tar
    tar xvf configs/${NAME}_configs.tar && rm configs/${NAME}_configs.tar
done
