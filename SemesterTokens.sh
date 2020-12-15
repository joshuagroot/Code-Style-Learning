#/bin/bash

assignments=($(find allSubmissions2NoComments -type d -name "*output"))

# echo $assignments
for assignmentFolder in ${assignments[@]}
do
    # echo $assignmentFolder

    subFiles=($(find $assignmentFolder -depth 1))

    for subFile in ${subFiles[@]}
    do
        # echo $subFile

        if [[ "$subFile" == *".txt" ]] ;then
            echo deleting
            rm $subFile
        fi
    done


    for subFile in ${subFiles[@]}
    do
        echo $subFile
        # if [[ "$subFile" == *".txt" ]] ;then
        #     echo deleting
        #     rm $subFile
        # fi
        if [[ "$subFile" == *"cpp" ]] || [[ "$subFile" == *"h" ]] ;then
             touch $subFile".txt"
             node Word2Vec/c-tokenizer/example/tokens.js < $subFile >> $subFile".txt"
             python Word2Vec/cleanTokens.py $subFile".txt"
        fi
    done

    # if [[ "$assignmentFolder" == *"assignment"* ]] ;then

    #     subFiles=($(find $assignmentFolder -depth 1))
    #     # echo ${subFiles[@]}
    #     for subFile in ${subFiles[@]}
    #     do 

    #         # echo $subFile

    #         if [[ "$subFile" == *".txt" ]] ;then
    #             echo deleting
    #             rm $subFile
    #         elif [[ "$subFile" == *"cpp" ]] || [[ "$subFile" == *"h" ]] ;then
    #              touch $subFile".txt"
    #              node c-tokenizer/example/tokens.js < $subFile >> $subFile".txt"
    #              python cleanTokens.py $subFile".txt"
    #         fi
    #     done  
    # fi
done