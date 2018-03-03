# script for extracting image file names and dimensions for specific objects

open INPUT1, "subsets/train_width_height.txt" or die $!;
open INPUT2, "../pascal/Annotations/ess/bicycle_train.ess" or die $!;
open OUTPUT, ">>subsets/train_width_height_bicycle.txt" or die $!;

while ($line1 = <INPUT1>) {
  $line2 = <INPUT2>;
  if ($line2 =~ /(\d{4}_\d{6}) (\d+).*/) {
    if ($2 > 0) {
      print OUTPUT $line1;
    }
  }
}

close INPUT1;
close INPUT2;
close OUTPUT;
