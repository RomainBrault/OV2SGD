#!/bin/sh
# mkdir build
# cd build
# CXX=/usr/local/gcc-6/bin/g++ cmake ..

# make -j8
# make install

fflag=no
cflag=no
mflag=no
iflag=no
jflag=1

optspec=":fcmij:-:"
while getopts "$optspec" optchar; do
    case "${optchar}" in
        f)
            fflag=yes
            ;;
        c)
            cflag=yes
            ;;
        m)
			mflag=yes
			;;
		i)
			iflag=yes
			;;
		j)
			jflag=$OPTARG
			;;
        *)
            if [ "$OPTERR" != 1 ] || [ "${optspec:0:1}" = ":" ]; then
                echo "Non-option argument: '-${OPTARG}'" >&2
            fi
            ;;
    esac
done

mkdir -p build
if [ $fflag = yes ] ; then rm -rf build ; mkdir build ; fi

if [ $cflag = yes ] ;
	then cd build .. ;
	cmake ..;
	cd ..;
fi

if [ $mflag = yes ] ; then cd build .. ; make -j${jflag}; cd ..; fi

if [ $iflag = yes ] ; then cd build .. ; make install; cd ..; fi