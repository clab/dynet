#!/usr/bin/perl -w

use strict;
use encoding 'utf8';

binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");
binmode(STDERR, ":utf8");


print "[+] Checking whether docker is running.\n";
my $service_status = `systemctl status docker.service`;
#print $service_status;

if($service_status =~ m/Active: inactive/) {
    print "[+] Docker was inactive, starting it now.\n";
    `sudo systemctl start docker.service`;
} else {
    print "[+] Docker is already active.\n";

}

1;
