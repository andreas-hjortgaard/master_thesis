# script for extracting image file names and dimensions for specific objects

open INPUT, "../cows-test/Annotations/TUcow_test.boxes" or die $!;
open OUTPUT, ">>../cows-test/Annotations/TUcow-test.ess" or die $!;

while ($line = <INPUT>) {
  if ($line =~ /(\S*) (.*)/) {
    print OUTPUT "$1 1 $2\n";
  }
}

close INPUT;
close OUTPUT;
