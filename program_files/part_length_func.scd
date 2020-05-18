~part_length_func = {
	o = rrand((~avg_part_length/~bit_duration) - (~avg_part_range/~bit_duration), (~avg_part_length/~bit_duration) + (~avg_part_range/~bit_duration)).round;
	if(o.isNegative == true, {o = o.neg;});
	if(o < (1/~bit_duration), {o = (1/~bit_duration);});
	o.value;
};