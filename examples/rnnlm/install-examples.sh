#!/bin/bash
set -e

data_version=ptb-mikolov-01.tar.gz

ROOTDIR=`dirname $0`
cd $ROOTDIR

rm -f $data_version
rm -rf ptb-mikolov
curl -f http://demo.clab.cs.cmu.edu/cdyer/$data_version -o $data_version
tar xzf $data_version
rm -f $data_version

echo SUCCESS. 1>&2

