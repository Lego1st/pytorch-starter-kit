if [[ ! -f "./datasets/data" ]]
then 
    mkdir -p ./datasets/data
fi

if [ -z "$1" ]; then
    echo "No data path supplied"
else
    ln -s $1/* ./datasets/data
fi