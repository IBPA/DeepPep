for i in $(seq 1 14) ; do
	echo -- $i --
	th experiment_3_multiArch.lua $i
done
