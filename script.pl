#!/usr/bin/perl

# This script is made to show graphs with git commit time made on workweek vs weekend
#
# The desription of this script and results of its usage is avaliable at:
# https://ivan.bessarabov.com/blog/famous-programmers-work-time-part-2-workweek-vs-weekend
#
# usage:
#
#   git log --author="Sebastian Riedel" --format="%H %ai" | perl script.pl
#


use strict;
use warnings FATAL => 'all';
use utf8;
use open qw(:std :utf8);
use feature qw(say);

use List::Util qw(max sum);
use Time::Local;

my %workweek;
my %weekend;

sub is_saturday_or_is_sunday {
    my ($yyyy_mm_dd) = @_;

        my ($year, $month, $day) = split /-/, $yyyy_mm_dd;

        my $timestamp = timegm(
                0,
                0,
                0,
                $day,
                $month - 1,
                $year,
        );

        my $wday = [gmtime($timestamp)]->[6];

        return $wday == 0 || $wday == 6;
}

while (my $line = <>) {

    # 181971ff7774853fceb0459966177d51eeab032c 2019-04-26 19:53:58 +0200

    my ($commit_hash, $date, $time, $timezone) = split / /, $line;
    my ($hour, $minute, $second) = split /:/, $time;

    $hour += 0;

    if (is_saturday_or_is_sunday($date)) {
        $weekend{$hour}++;
    } else {
        $workweek{$hour}++;
    }
}

my $max = max(values(%workweek), values(%weekend));

my $format = "%6s   %6s %-30s  %6s %-30s",

say '';
say sprintf $format, 'hour', '', 'Monday to Friday', '', 'Saturday and Sunday';

foreach my $hour (0..23) {
    $workweek{$hour} //= 0;
    $weekend{$hour} //= 0;
    say sprintf $format,
        sprintf('%02d', $hour),
        $workweek{$hour},
        '*' x ($workweek{$hour} / $max * 25),

        $weekend{$hour},
        '*' x ($weekend{$hour} / $max * 25),
        ;
}

my $total_commits_workweek = sum(values %workweek);
my $total_commits_weekend = sum(values %weekend);
my $total_commits = $total_commits_workweek + $total_commits_weekend;

say '';
say sprintf $format,
    'Total:',
    $total_commits_workweek,
    sprintf('(%.1f%%)', $total_commits_workweek * 100 / $total_commits),
    $total_commits_weekend,
    sprintf('(%.1f%%)', $total_commits_weekend* 100 / $total_commits),
    ;

say '';