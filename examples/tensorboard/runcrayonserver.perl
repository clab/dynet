#!/usr/bin/perl -w

# This script makes the assumption that there is only one image with name alband/crayon, which should always be true, but also it assumes the crayon container name is crayon and that there is only one docker container running from the already mentioned image.


use strict;
use encoding 'utf8';

binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");
binmode(STDERR, ":utf8");

my ($p1, $p2) = @ARGV;

$p1 = 8888 if not defined($p1);
$p2 = 8889 if not defined($p2);

print "[+] Checking for docker images.\n";
my $docker_images = `sudo docker images`;

if($docker_images =~ m/alband\/crayon/){
    print "Docker image already exists. If for some reason you want to pull a different version you have to do it manually\n";    
} else {
    print "[+] Downloading docker image.\n";
    `sudo docker pull alband/crayon`;
}

print "[+] Checking for docker containers.\n";
my $docker_containers = `sudo docker ps -a`;

if($docker_containers =~ m/alband\/crayon/){
    print "[+] Restarting docker\n";
    `sudo docker restart crayon`;    
} else {
    print "[+] Docker container does not exist,  running for the first time.";
    my $dock_run = "sudo docker run -d -p $p1:$p1 -p $p2:$p2 --name crayon alband/crayon";
    print "Running $dock_run\n";
    `$dock_run`;
}


1;
