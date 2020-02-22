if [[ ! -f "./data" ]]
then 
    mkdir ./data
fi

if [ -z "$1" ]; then
    echo "No data path supplied"
else
    ln -s $1/* data/
fi