#!/bin/bash

./entitynlm --dynet-seed 1234 \
	    --dynet-mem 256 \
	    --task train \
	    --modeltype dis \
	    --trnfile data/dickens/data/trn-oliver.txt \
	    --devfile data/dickens/data/trn-oliver.txt \
	    --mlen 4 \
	    --evalstep 1 \
	    --path dickens
