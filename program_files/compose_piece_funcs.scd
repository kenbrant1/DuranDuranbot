// BUILD/GENERATE/COMPOSE PIECE

~comp_machine_learning = 1; //composition process machine learning is on (default)


~ml_0to12_array = (0..12);
~interval_array = (1..12);
~comp_process_interval_array = (0..11);
~bit_choice_array = [1, 2, 4, 8, 16, 32, 64, 128];
s = ~bit_duration;
~time_array = [~bit_duration.value]++Array.fill(~bit_choice_array.size - 1, {s = s * 2});
~bit_choice_array_num = (1..~bit_choice_array.size);
~blend_choice_array = [(-6),(-5),(-4),(-3),(-2),(-1),0,1,2,3,4,5,6];

~midi_low_limit = 36;
~midi_high_limit = 60;
~freqmap = (i = ~tuning_frequency / (2 ** (69 / 12)); a = (-1); b = (-1); Array.fill(128, {i * (2 ** (((a=a+1) / 12).trunc)) * ( (2 ** (12.reciprocal)) ** ((b=b+1)  % 12));}));/*~freqmap = (i = 8.0271480262684/*-0.0000000000024*/; ~stepratio = 1.0594630943593; Array.fill(128, { i = i*~stepratio;}));*/
~freqmap = ~freqmap.cpsmidi;
/*~freqmap2 = (i = ~freqmap.at(58)/*-0.0000000000024*/; ~stepratio = 1.0594630943593; Array.fill(24, { i = i*~stepratio;}));*/
~freqmap2 = (n = ~midi_low_limit-1; Array.fill(~midi_high_limit-~midi_low_limit+1, {~freqmap.at(n=n+1);}));
/*~ratio_array = (x = 0; Array.fill(12, {1.0594630943593**(x=x+1)}));*/
~interval_array = (1..12);
~midi_ratio_array = (1..12);

~part_length_func = {
	o = rrand((~avg_part_length/~bit_duration) - (~avg_part_range/~bit_duration), (~avg_part_length/~bit_duration) + (~avg_part_range/~bit_duration)).round;
	if(o.isNegative == true, {o = o.neg;});
	if(o < (1/~bit_duration), {o = (1/~bit_duration);});
	o.value;
};

~clock_diff_func = {

	~clock_diff = ~clock_diff_var_1 - ~clock_diff_var_2;
	if(~clock_diff.isFloat == true, {~clock_diff = ~clock_diff.round});
	if(~clock_diff.isNegative == true, {~clock_diff = ~clock_diff.neg});
	while({~clock_diff > 12}, {~clock_diff = ~clock_diff - 12});
	~clock_diff.value;

};

~note_generator_func = {

	if(~number_of_instruments == 1, {
		if(("old_instrument_"++(~inst_num)++"_freq_1").asSymbol.envirGet != nil, {
			("old_instrument_"++(~inst_num)++"_freq_2").asSymbol.envirPut(("old_instrument_"++(~inst_num)++"_freq_1").asSymbol.envirGet);
		}, {
			("old_instrument_"++(~inst_num)++"_freq_2").asSymbol.envirPut(("instrument_"++(~inst_array.wrapAt(~inst_num))++"_freq").asSymbol.envirGet);
		});
	});

	("old_instrument_"++(~inst_num)++"_freq_1").asSymbol.envirPut(("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet);

	/*case
	{~number_of_instruments == 1}{1.do(s = (-1); {("clock_diff_"++(s=s+1)++"_func").asSymbol.envirGet.value;});}
	{~number_of_instruments == 2}{2.do(s = (-1); {("clock_diff_"++(s=s+1)++"_func").asSymbol.envirGet.value;});}
	{~number_of_instruments == 3}{3.do(s = (-1); {("clock_diff_"++(s=s+1)++"_func").asSymbol.envirGet.value;});};*/

	/*	(~number_of_instruments-1).do(u = (-1); {
	u=u+1;
	("clock_diff_"++(u)).asSymbol.envirPut(("clock_diff_"++(u)).asSymbol.envirGet.round.asInteger;);
	});*/

	case
	{~number_of_instruments == 1}{
		~clock_diff_var_1 = ("instrument_"++(~inst_array.wrapAt(~inst_num-1))++"_freq").asSymbol.envirGet;
		~clock_diff_var_2 = ("old_instrument_"++(~inst_num)++"_freq_1").asSymbol.envirGet;
		~clock_diff_func.value;
		~clock_diff_1 = ~clock_diff.value;

		~clock_diff_var_1 = ("old_instrument_"++(~inst_num)++"_freq_1").asSymbol.envirGet;
		~clock_diff_var_2 = ("old_instrument_"++(~inst_num)++"_freq_2").asSymbol.envirGet;
		~clock_diff_func.value;
		~clock_diff_2 = ~clock_diff.value;
	}
	{~number_of_instruments == 2}{
		~clock_diff_var_1 = ("instrument_"++(~inst_array.wrapAt(~inst_num-1))++"_freq").asSymbol.envirGet;
		~clock_diff_var_2 = ("old_instrument_"++(~inst_num)++"_freq_1").asSymbol.envirGet;
		~clock_diff_func.value;
		~clock_diff_1 = ~clock_diff.value;

		~clock_diff_var_1 = ("instrument_"++(~inst_array.wrapAt(~inst_num-1))++"_freq").asSymbol.envirGet;
		~clock_diff_var_2 = ("instrument_"++(~inst_array.wrapAt(~inst_num))++"_freq").asSymbol.envirGet;
		~clock_diff_func.value;
		~clock_diff_2 = ~clock_diff.value;
	}
	{~number_of_instruments == 3}{
		~clock_diff_var_1 = ("instrument_"++(~inst_array.wrapAt(~inst_num-1))++"_freq").asSymbol.envirGet;
		~clock_diff_var_2 = ("old_instrument_"++(~inst_num)++"_freq_1").asSymbol.envirGet;
		~clock_diff_func.value;
		~clock_diff_1 = ~clock_diff.value;

		~clock_diff_var_1 = ("instrument_"++(~inst_array.wrapAt(~inst_num-1))++"_freq").asSymbol.envirGet;
		~clock_diff_var_2 = ("instrument_"++(~inst_array.wrapAt(~inst_num))++"_freq").asSymbol.envirGet;
		~clock_diff_func.value;
		~clock_diff_2 = ~clock_diff.value;

		~clock_diff_var_1 = ("instrument_"++(~inst_array.wrapAt(~inst_num-1))++"_freq").asSymbol.envirGet;
		~clock_diff_var_2 = ("instrument_"++(~inst_array.wrapAt(~inst_num+1))++"_freq").asSymbol.envirGet;
		~clock_diff_func.value;
		~clock_diff_3 = ~clock_diff.value;
	};

	~intv_weights = ("ml_data_array_"++(~number_of_instruments.asString)++"_insts_"++((~player_array.wrapAt(~inst_num-1)))).asSymbol.envirGet.at(
		~ml_data_array_num = case
		{~number_of_instruments <= 2}{~clock_diff_1*(~interval_array.size+1)+~clock_diff_2}
		{~number_of_instruments == 3}{~clock_diff_1*(~interval_array.size+1)*(~interval_array.size+1)+(~clock_diff_2*(~interval_array.size+1))+~clock_diff_3};
	).normalizeSum;

	~new_interval = ~interval_array.wchoose(~intv_weights);
	~new_interval = [~new_interval, ~new_interval.neg].choose;

	("instrument_"++(~inst_num)++"_freq").asSymbol.envirPut(("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet + ~new_interval);

	if(((("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet > ~midi_high_limit) or: (("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet < ~midi_low_limit)), {
		while({("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet > ~midi_high_limit}, {("instrument_"++(~inst_num)++"_freq").asSymbol.envirPut(("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet-12)});
		while({("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet < ~midi_low_limit}, {("instrument_"++(~inst_num)++"_freq").asSymbol.envirPut(("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet+12)});
	});

	("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet;
};

~time_generator_func = {

	(~number_of_instruments).do(e = 0; {
		e=e+1;
		case
		{(("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(0)) and: (("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(1))}{if((("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(0)+~time_array.at(1)/2)), {("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(0)}, {("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(1)})}
		{(("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(1)) and: (("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(2))}{if((("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(1)+~time_array.at(2)/2)), {("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(1)}, {("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(2)})}
		{(("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(2)) and: (("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(3))}{if((("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(2)+~time_array.at(3)/2)), {("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(2)}, {("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(3)})}
		{(("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(3)) and: (("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(4))}{if((("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(3)+~time_array.at(4)/2)), {("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(3)}, {("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(4)})}
		{(("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(4)) and: (("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(5))}{if((("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(4)+~time_array.at(5)/2)), {("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(4)}, {("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(5)})}
		{(("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(5)) and: (("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(6))}{if((("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(5)+~time_array.at(6)/2)), {("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(5)}, {("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(6)})}
		{(("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(6)) and: (("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(7))}{if((("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(6)+~time_array.at(7)/2)), {("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(6)}, {("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(7)})}
		{(("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(7))}{("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(7)};
	});

	if(~number_of_instruments == 1, {
		(~number_of_instruments).do(e = 0; {
			e=e+1;
			case
			{(("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration >= ~time_array.at(0)) and: (("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration <= ~time_array.at(1))}{if((("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration < (~time_array.at(0)+~time_array.at(1)/2)), {("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place_1").asSymbol.envirPut(0)}, {("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place_1").asSymbol.envirPut(1)})}
			{(("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration >= ~time_array.at(1)) and: (("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration <= ~time_array.at(2))}{if((("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration < (~time_array.at(1)+~time_array.at(2)/2)), {("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place_1").asSymbol.envirPut(1)}, {("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place_1").asSymbol.envirPut(2)})}
			{(("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration >= ~time_array.at(2)) and: (("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration <= ~time_array.at(3))}{if((("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration < (~time_array.at(2)+~time_array.at(3)/2)), {("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place_1").asSymbol.envirPut(2)}, {("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place_1").asSymbol.envirPut(3)})}
			{(("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration >= ~time_array.at(3)) and: (("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration <= ~time_array.at(4))}{if((("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration < (~time_array.at(3)+~time_array.at(4)/2)), {("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place_1").asSymbol.envirPut(3)}, {("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place_1").asSymbol.envirPut(4)})}
			{(("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration >= ~time_array.at(4)) and: (("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration <= ~time_array.at(5))}{if((("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration < (~time_array.at(4)+~time_array.at(5)/2)), {("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place_1").asSymbol.envirPut(4)}, {("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place_1").asSymbol.envirPut(5)})}
			{(("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration >= ~time_array.at(5)) and: (("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration <= ~time_array.at(6))}{if((("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration < (~time_array.at(5)+~time_array.at(6)/2)), {("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place_1").asSymbol.envirPut(5)}, {("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place_1").asSymbol.envirPut(6)})}
			{(("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration >= ~time_array.at(6)) and: (("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration <= ~time_array.at(7))}{if((("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration < (~time_array.at(6)+~time_array.at(7)/2)), {("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place_1").asSymbol.envirPut(6)}, {("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place_1").asSymbol.envirPut(7)})}
			{(("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice_1").asSymbol.envirGet*~bit_duration >= ~time_array.at(7))}{("old_bit_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place_1").asSymbol.envirPut(7)};
		});
	});


	/*
	("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+2))++"_choice").asSymbol.envirGet
	("bit_clock_"++(~inst_array.wrapAt(~inst_num-1+2))++"_place").asSymbol.envirGet
	*/

	case
	{~number_of_instruments == 1}{
		~length_comp_1 = ("bit_clock_"++(~inst_array.wrapAt(~inst_num-1))++"_place").asSymbol.envirGet;
		~length_comp_2 = ("old_bit_clock_"++(~inst_num)++"_place_1").asSymbol.envirGet;

	}
	{~number_of_instruments == 2}{
		~length_comp_1 = ("bit_clock_"++(~inst_array.wrapAt(~inst_num-1))++"_place").asSymbol.envirGet;
		~length_comp_2 = ("bit_clock_"++(~inst_array.wrapAt(~inst_num))++"_place").asSymbol.envirGet;

	}
	{~number_of_instruments == 3}{
		~length_comp_1 = ("bit_clock_"++(~inst_array.wrapAt(~inst_num-1))++"_place").asSymbol.envirGet;
		~length_comp_2 = ("bit_clock_"++(~inst_array.wrapAt(~inst_num))++"_place").asSymbol.envirGet;
		~length_comp_3 = ("bit_clock_"++(~inst_array.wrapAt(~inst_num+1))++"_place").asSymbol.envirGet;

	};

	~time_weights = ("ml_data_array_"++(~number_of_instruments.asString)++"_insts_"++((~player_array.wrapAt(~inst_num-1)))).asSymbol.envirGet.at(
		~ml_data_array_num = case
		{~number_of_instruments <= 2}{~length_comp_1*(~time_array.size+1)+~length_comp_2}
		{~number_of_instruments == 3}{~length_comp_1*(~time_array.size+1)*(~time_array.size+1)+(~length_comp_2*(~time_array.size+1))+~length_comp_3};
	).normalizeSum;

	~new_time = ~bit_choice_array.wchoose(~time_weights);

	if(~number_of_instruments == 1, {
		("old_bit_clock_"++(~inst_num)++"_choice_1").asSymbol.envirPut(("bit_clock_"++(~inst_num)++"_choice").asSymbol.envirGet);
	});

	("bit_clock_"++(~inst_num)++"_choice").asSymbol.envirPut(~new_time);
};

~inst_clock_func = {

	~number_of_instruments.do(n = 0; {("bit_clock_"++(n=n+1)).asSymbol.envirPut(("bit_clock_"++(n)).asSymbol.envirGet+1);});
	~number_of_instruments.do(~inst_num = 0; {
		~inst_num = ~inst_num + 1;
		("instrument_"++(~inst_num)++"_array").asSymbol.envirPut(
			("instrument_"++(~inst_num)++"_array").asSymbol.envirGet.add(
				if(("bit_clock_"++(~inst_num)).asSymbol.envirGet == ("bit_clock_"++(~inst_num)++"_choice").asSymbol.envirGet, {
					("bit_clock_"++(~inst_num)).asSymbol.envirPut((-1));
					("instrument_"++(~inst_num)++"_if_zero").asSymbol.envirPut("false");
					~note_generator_func.value; //find new interval for current instrument
					~time_generator_func.value; //find new bit duration for current instrument
					~put1 = ("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet;
				}, {
					~put1 = if(("instrument_"++(~inst_num)++"_if_zero").asSymbol.envirGet == "false", {("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet;}, {0});
				});
				~put1.value;
			);
		);
	});
};


~piece_generator_func = {

	// THERE IS A POINT TO USING BIT-BY-BIT CIRCULAR COMPOSITION ( https://scsynth.org/t/bit-by-bit-circular-composition/1107 )
	// Not using it, DDbot player can't compose based on everyone else, for the player isn't composing in parallel.
	// Players NEED to be composing in parallel with everyone else... On the "same track".
	//...... It's also NEEDED for machine learning functions.

	(
		(

			"gen checkpoint 1".post;

			thisThread.randSeed = ~randseed_1;

			~player_array = (x = 0; Array.fill(if(FileReader.read((~ddbot_root)++"/program_files/other_program_files/which_players.scd", false, true).flatten.at(0) == 0.asString, {~number_of_instruments}, {1}), {"player_"++(x=x+1)}));
			~player_array = ~player_array.scramble;
			~number_of_instruments.do(x = 0; {("inst_"++(x=x+1)).asSymbol.envirPut(~player_array.at(x-1));});


			x = 0;
			~inst_array = Array.fill(~number_of_instruments, {x=x+1});



			if(~duranduranbot_routine.isPlaying == true, {~number_of_instruments.do(x = 0; {("duranduranbotsynth"++(x=x+1)).asSymbol.envirGet.set(\freq, 0);});});


			~number_of_instruments.do(x = 0; {
				x=x+1;
				~number_of_parts_in_piece.do(y = 0; {("instrument_"++(x)++"_array_part_"++(y=y+1)).asSymbol.envirPut([]);});
			});

		);

		" 2".post;

		//Piece creation

		(

			~create_piece_function = {

				// get ml data arrays

				~player_array.size.do(~inst_num = 0; {
					~inst_num=~inst_num+1;
					("ml_data_array_"++(~number_of_instruments.asString)++"_insts_"++((~player_array.wrapAt(~inst_num-1)))).asSymbol.envirPut(File.readAllString((~ddbot_root)++"/program_files/training_data_folder/ddbot_training_data/"++(~number_of_instruments.asString)++"_inst_"++(~player_array.wrapAt(~inst_num-1))++"_interval_weights.scd", r).interpret);
				});
				~player_array.size.do(~inst_num = 0; {
					~inst_num=~inst_num+1;
					("ml_data_array_"++(~number_of_instruments.asString)++"_insts_"++((~player_array.wrapAt(~inst_num-1)))).asSymbol.envirPut(File.readAllString((~ddbot_root)++"/program_files/training_data_folder/ddbot_training_data/"++(~number_of_instruments.asString)++"_inst_"++(~player_array.wrapAt(~inst_num-1))++"_time_weights.scd", r).interpret);
				});
				~ml_data_array = File.readAllString((~ddbot_root)++"/program_files/training_data_folder/ddbot_training_data/compose_ml_data.scd", r).interpret;

				~number_of_instruments.do(x = 0; {
					x=x+1;
					("instrument_"++(x)++"_full").asSymbol.envirPut([]);
					("instrument_"++(x)++"_array").asSymbol.envirPut([]);
				});


				thisThread.randSeed = ~randseed_2;


				// choose compoition process

				~composition_process_setting = 1;

				case
				{~composition_process_setting == 1}{

					case
					{~ai_assist_compose != 1}{

						// I NEED DDBOT TO SOUND *KINDA ROBOTIC, AND KINDA HUMAN*


						/*

						HOW DDBOT COMPOSITION WORKS -

						1. Generate two sequences of notes/chords using DDbot's machine learning data, using the "~inst_clock_func"
						function. Then "blend" those two sequences by overlaying each other, the 1st sequence running fowards,
						and the 2nd sequence running reverse... and "blend" them by finding the average of the overlayed
						two notes by adding them together and diving by 2.

						2. Then "copy" that sequence and create a larger "sequence" by repeating that sequence a number of times.

						3. For each note/chord of this larger sequence, and generate it's own "sub-sequence" using DDbot's
						machine learning data using the "~inst_clock_func" function... and "blend"(how this is done is described
						at the * below) that sequence with the sequence generated by the note/chord after the chord which was just
						generated from, using the same blend method used before.

						4. Repeat this for all chords in the original "top-level" sequnce".

						^ The reason for doing this is to provide "relation" between every part in the entire piece of music.

						* use machine learning to "blend" the two sequences by using the interval of the two notes that are to be
						blended, along with the interval of the two notes which were blended prior to those. To "blend" is to
						find the sum of the the two notes then divide that sum by 2.... and then add a value in the ~blend_choice_array
						using machine learning data.

						*/

						// HAS BEEN TESTED TO WORK

						~number_of_instruments.do(x = 0; {
							x=x+1;
							("top_level_notes_"++(x)).asSymbol.envirPut([]);
						});

						//generate first set of notes

						~number_of_instruments.do(n = 0; {
							~inst_num = (n=n+1);
							("top_level_notes_"++(~inst_array.wrapAt(~inst_num-1))).asSymbol.envirPut([rrand(~midi_low_limit, ~midi_high_limit)]);
							("instrument_"++(~inst_array.wrapAt(~inst_num-1))++"_freq").asSymbol.envirPut(("top_level_notes_"++(~inst_array.wrapAt(~inst_num-1))).asSymbol.envirGet.at(0).asInteger);
						});

						/*n = 3; x = Array.fill(13, {n=n+1});
						y = Array.fill(x.size, {1/x.size}).normalizeSum;
						z = (x.wchoose(y.value)-1);
						(if(z.odd == true, {[z=z+[1,(-1)].choose, z].wchoose([2/3,1/3].normalizeSum)}))*/

						//generate string of notes using cpmposition process

						(exprand(4, 16).round-1).do({
							~number_of_instruments.do(n = 0; {
								~inst_num = (n=n+1);
								~note_generator_func.value;
								("top_level_notes_"++(~inst_array.wrapAt(~inst_num-1))).asSymbol.envirPut(("top_level_notes_"++(~inst_array.wrapAt(~inst_num-1))).asSymbol.envirGet.add(("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet;););
							});
						});

						~composition_setting = 2;

						if(~composition_setting == 2, {
							// blend

							/*("top_level_notes_"++(1)).asSymbol.envirGet
							("top_level_notes_"++(1)++"_forwards").asSymbol.envirGet
							("top_level_notes_"++(1)++"_reverse").asSymbol.envirGet*/

							~number_of_instruments.do(n = 0; {("top_level_notes_"++(n=n+1)++"_forwards").asSymbol.envirPut(("top_level_notes_"++(n)).asSymbol.envirGet)});
							~number_of_instruments.do(n = 0; {("top_level_notes_"++(n=n+1)++"_reverse").asSymbol.envirPut(("top_level_notes_"++(n)).asSymbol.envirGet.reverse)});

							~number_of_instruments.do(n = 0; {n=n+1; ("top_level_notes_"++(n)++"_blended").asSymbol.envirPut([]);});

							~number_of_instruments.do(n = 0; {
								n=n+1;
								(("top_level_notes_"++(1)++"_forwards").asSymbol.envirGet.size).do(i = (-1); {
									i=i+1;
									u = ((("top_level_notes_"++(n)++"_forwards").asSymbol.envirGet.at(i) + ("top_level_notes_"++(n)++"_reverse").asSymbol.envirGet.at(i)) / 2).round;
									("top_level_notes_"++(n)++"_blended").asSymbol.envirPut(("top_level_notes_"++(n)++"_blended").asSymbol.envirGet.add(u.value));
								});
							});
							~number_of_instruments.do(n = 0; {("top_level_notes_"++(n=n+1)).asSymbol.envirPut(("top_level_notes_"++(n)++"_blended").asSymbol.envirGet);});
						});

						// iterate sequence
						y = rrand(2, 8);
						~number_of_instruments.do(x = 0; {
							("top_level_notes_"++(x=x+1)++"_full").asSymbol.envirPut([]);
							y.do({
								("top_level_notes_"++(x)++"_full").asSymbol.envirPut(("top_level_notes_"++(x)++"_full").asSymbol.envirGet.add(("top_level_notes_"++(x)).asSymbol.envirGet;));
							});
							("top_level_notes_"++(x)++"_full").asSymbol.envirPut(("top_level_notes_"++(x)++"_full").asSymbol.envirGet.flatten);
						});



						/*
						("top_level_notes_"++(3)++"_full").asSymbol.envirGet
						~top_level_notes_1.postln;
						~top_level_notes_2.postln;
						~top_level_notes_3.postln;

						("instrument_"++(1)++"_freq").asSymbol.envirGet
						*/


						// generate bottom-level chords/piece

						thisThread.randSeed = ~randseed_3;

						~number_of_instruments.do(n = 0; {("bit_clock_"++(n=n+1)).asSymbol.envirPut((-1));});
						~number_of_instruments.do(n = 0; {("bit_clock_"++(n=n+1)++"_choice").asSymbol.envirPut(~bit_choice_array.choose)});
						~number_of_instruments.do(n = 0; {("old_bit_clock_"++(n=n+1)++"_choice_1").asSymbol.envirPut(("bit_clock_"++(n)++"_choice").asSymbol.envirGet)});
						~number_of_instruments.do(x = 0; {("instrument_"++(x=x+1)++"_if_zero").asSymbol.envirPut("false");});
						/*~number_of_instruments.do(n = 0; {("bottom_level_notes_"++(n=n+1)).asSymbol.envirPut([])});*/

						~number_of_instruments.do(n = 0; {("instrument_"++(n=n+1)++"_full").asSymbol.envirPut([]);});

						~first_comp_part_func = {

							// "put" top level chords into ~instrument_n_freq's

							~number_of_instruments.do(n = 0; {
								n=n+1;
								("instrument_"++(n)++"_freq").asSymbol.envirPut(("top_level_notes_"++(n)++"_full").asSymbol.envirGet.at(x).asInteger);
							});

							// run composition process with times/durations

							~number_of_instruments.do(n = 0; {
								n=n+1;
								("instrument_"++(n)++"_array").asSymbol.envirPut([]);
								("instrument_"++(n)++"_array_1").asSymbol.envirPut([]);
							});
							(~part_length_func.value).do({~inst_clock_func.value});
							~number_of_instruments.do(n = 0; {("instrument_"++(n=n+1)++"_array_1").asSymbol.envirPut(("instrument_"++(n)++"_array").asSymbol.envirGet)});

						};


						//compose action

						~composition_setting = 2;
						~number_of_instruments.do(n = 0; {("note_1_inst_"++(n=n+1)).asSymbol.envirPut(nil);});
						~number_of_instruments.do(n = 0; {("note_2_inst_"++(n=n+1)).asSymbol.envirPut(nil);});
						~number_of_instruments.do(n = 0; {("comp_int_2_inst_"++(n=n+1)).asSymbol.envirPut(0);});
						~number_of_instruments.do(n = 0; {("instrument_"++(n=n+1)++"comp_interval_array").asSymbol.envirPut([]);});
						~number_of_instruments.do(n = 0; {("instrument_"++(n=n+1)++"comp_choice_array").asSymbol.envirPut([]);});

						(~top_level_notes_1_full.size-1).do(x = (-1); ~part_num = 0; {
							x=x+1;
							~part_num = ~part_num + 1;


							if(~composition_setting == 1, {~first_comp_part_func.value}, {
								if(x > 0, {

									~number_of_instruments.do(n = 0; {
										n=n+1;
										("instrument_"++(n)++"_array_1").asSymbol.envirPut(
											case
											{~composition_setting == 2}{("instrument_"++(n)++"_array_2_forwards").asSymbol.envirGet;}
											{~composition_setting == 3}{("instrument_"++(n)++"_array_blended").asSymbol.envirGet;};
										);
									});
								}, {~first_comp_part_func.value});
							});


							~number_of_instruments.do(n = 0; {
								n=n+1;
								("instrument_"++(n)++"_freq").asSymbol.envirPut(("top_level_notes_"++(n)++"_full").asSymbol.envirGet.at(x+1).asInteger);
							});

							~number_of_instruments.do(n = 0; {
								n=n+1;
								("instrument_"++(n)++"_array").asSymbol.envirPut([]);
								("instrument_"++(n)++"_array_2").asSymbol.envirPut([]);
							});
							o.do({~inst_clock_func.value});
							~number_of_instruments.do(n = 0; {("instrument_"++(n=n+1)++"_array_2").asSymbol.envirPut(("instrument_"++(n)++"_array").asSymbol.envirGet)});

							// reverse array 2
							~number_of_instruments.do(n = 0; {("instrument_"++(n=n+1)++"_array_2_forwards").asSymbol.envirPut(("instrument_"++(n)++"_array_2").asSymbol.envirGet)});
							~number_of_instruments.do(n = 0; {("instrument_"++(n=n+1)++"_array_2").asSymbol.envirPut(("instrument_"++(n)++"_array_2").asSymbol.envirGet.reverse)});

							~number_of_instruments.do(n = 0; {n=n+1; ("instrument_"++(n)++"_array_blended").asSymbol.envirPut([]);});


							// DEVELOP WAY TO FOR AI TO HELP COMPOSE.
							// LET HUMAN USER EXPERIMENT WITH EACH COMPOSING SECTION.
							//


							// blend, and use ml data
							// ~comp_int2 equals 0 at very beginning of composition process
							~number_of_instruments.do(n = 0; {var comp_choice;
								n=n+1;
								o.do(i = (-1); {
									i=i+1;
									if(~comp_machine_learning == 1, {

										if(((("instrument_"++(n)++"_array_1").asSymbol.envirGet.at(i) != ("note_1_inst_"++(n)).asSymbol.envirGet) or: (("instrument_"++(n)++"_array_2").asSymbol.envirGet.at(i) != ("note_2_inst_"++(n)).asSymbol.envirGet)), {

											("comp_result_"++(n)).asSymbol.envirPut(
												(((("note_1_inst_"++(n)).asSymbol.envirPut(
													("instrument_"++(n)++"_array_1").asSymbol.envirGet.at(i)
												).value

												)
												+ (("note_2_inst_"++(n)).asSymbol.envirPut(
													("instrument_"++(n)++"_array_2").asSymbol.envirGet.at(i)
												).value)) / 2).round;

											);

											("comp_int_1_inst_"++(n)).asSymbol.envirPut(
												("note_1_inst_"++(n)).asSymbol.envirGet - ("note_2_inst_"++(n)).asSymbol.envirGet

											);

											if(("comp_int_1_inst_"++(n)).asSymbol.envirGet.isNegative == true, {
												("comp_int_1_inst_"++(n)).asSymbol.envirPut(
													("comp_int_1_inst_"++(n)).asSymbol.envirGet.neg

												)
											});

											while({("comp_int_1_inst_"++(n)).asSymbol.envirGet >= 12}, {
												("comp_int_1_inst_"++(n)).asSymbol.envirPut(
													("comp_int_1_inst_"++(n)).asSymbol.envirGet - 12

												)
											});

											~comp_choice = ~blend_choice_array.wchoose(
												~ml_data_array.at(
													("comp_int_1_inst_"++(n)).asSymbol.envirGet.value*(~blend_choice_array.size-1)+("comp_int_2_inst_"++(n)).asSymbol.envirGet.value;

												).normalizeSum;

											);

											("comp_result_"++(n)).asSymbol.envirPut(
												("comp_result_"++(n)).asSymbol.envirGet + ~comp_choice.value

											);

											if(((("comp_result_"++(n)).asSymbol.envirGet > ~midi_high_limit) or: (("comp_result_"++(n)).asSymbol.envirGet < ~midi_low_limit)), {
												while({("comp_result_"++(n)).asSymbol.envirGet > ~midi_high_limit}, {("comp_result_"++(n)).asSymbol.envirPut(("comp_result_"++(n)).asSymbol.envirGet-12)});
												while({("comp_result_"++(n)).asSymbol.envirGet < ~midi_low_limit}, {("comp_result_"++(n)).asSymbol.envirPut(("comp_result_"++(n)).asSymbol.envirGet+12)});
											});

											("comp_int_2_inst_"++(n)).asSymbol.envirPut(
												("comp_int_1_inst_"++(n)).asSymbol.envirGet

											);
										});

										("instrument_"++(n)++"_array_blended").asSymbol.envirPut(("instrument_"++(n)++"_array_blended").asSymbol.envirGet.add(("comp_result_"++(n)).asSymbol.envirGet.value));

										("instrument_"++(n)++"comp_interval_array").asSymbol.envirPut(("instrument_"++(n)++"comp_interval_array").asSymbol.envirGet.add(("comp_int_1_inst_"++(n)).asSymbol.envirGet));

										("instrument_"++(n)++"comp_choice_array").asSymbol.envirPut(("instrument_"++(n)++"comp_choice_array").asSymbol.envirGet.add(~comp_choice.value));

									}, {
										u = ((("instrument_"++(n)++"_array_1").asSymbol.envirGet.at(i) + ("instrument_"++(n)++"_array_2").asSymbol.envirGet.at(i)) / 2).round;

										("instrument_"++(n)++"_array_blended").asSymbol.envirPut(("instrument_"++(n)++"_array_blended").asSymbol.envirGet.add(u.value));

									});
								});
							});

							~number_of_instruments.do(n = 0; {
								n=n+1;
								("instrument_"++(n)++"_full").asSymbol.envirPut(("instrument_"++(n)++"_full").asSymbol.envirGet.add(("instrument_"++(n)++"_array_blended").asSymbol.envirGet;));
								("instrument_"++(n)++"_full").asSymbol.envirPut(("instrument_"++(n)++"_full").asSymbol.envirGet.flatten);
							});

						});


						/*	("instrument_"++(2)++"_full").asSymbol.envirGet;*/


						/*	("instrument_"++(1)++"_array_2").asSymbol.envirGet;
						("instrument_"++(2)++"_array_2").asSymbol.envirGet;
						("instrument_"++(3)++"_array_2").asSymbol.envirGet;

						("instrument_"++(1)++"_array_2").asSymbol.envirGet.size;
						("instrument_"++(2)++"_array_2").asSymbol.envirGet.size;
						("instrument_"++(3)++"_array_2").asSymbol.envirGet.size;
						*/

					}
					{~ai_assist_compose == 1}{
					};

				}
				{~composition_process_setting == 2}{

					thisThread.randSeed = ~randseed_2;

					~part_bit_size = 64;
					~max_number_of_parts = 200; //max 200 parts. random probability choice decreases linearly from "~max_number_of_parts" value to 1.
					~number_of_parts_in_piece = (~max_number_of_parts.linrand - ~max_number_of_parts).neg;
					~just_ml_piece_size = ~part_bit_size*~number_of_parts_in_piece;

					thisThread.randSeed = ~randseed_3;

					~number_of_instruments.do(y = 0; {("bit_clock_"++(y=y+1)).asSymbol.envirPut(0);});


					//generation of piece

					~number_of_instruments.do(x = 0; {("instrument_"++(x=x+1)++"_if_zero").asSymbol.envirPut("false");});

					~number_of_instruments.do(x = 0; {
						x=x+1;
						("instrument_"++(x)++"_array").asSymbol.envirPut(
							("instrument_"++(x)++"_array").asSymbol.envirGet.add((("instrument_"++(x)++"_freq").asSymbol.envirPut(~freqmap2.choose)).value)
						);
						("bit_clock_"++(x)++"_choice").asSymbol.envirPut(~bit_choice_array.choose);
					});

					~number_of_instruments.do(s = 0; {("old_instrument_"++(s=s+1)++"_freq_1").asSymbol.envirPut(("instrument_"++(s)++"_array").asSymbol.envirGet.at(0).asInteger + 12)});

					/*
					("bit_clock_"++(3)++"_choice").asSymbol.envirGet*/

					/*(~number_of_instruments-1).do(x = 1; {
					("instrument_"++(x=x+1)++"_freq").asSymbol.envirPut(0);
					("bit_clock_"++(x)++"_choice").asSymbol.envirPut(0);/*
					("bit_clock_"++(2)++"_choice").asSymbol.envirGet <~time_array.at(0)           (0);*/
					});*/

					/*(~number_of_instruments-1).do(s = 0; {("bit_clock_"++(s=s+1)).asSymbol.envirPut(("bit_clock_"++(s)).asSymbol.envirGet+1);});
					(~number_of_instruments-1).do(s = 1; {
					s=s+1;
					~bit_clock = ("bit_clock_"++(s)).asSymbol.envirGet;
					~inst_num = s;
					~inst_clock_func.value;
					});*/

					(~just_ml_piece_size-1).do({~inst_clock_func.value;});

					~number_of_instruments.do(n = 0; {
						n=n+1;
						("instrument_"++(n)++"_full").asSymbol.envirPut(("instrument_"++(n)++"_array").asSymbol.envirGet;);
						("instrument_"++(n)++"_array").asSymbol.envirPut([]);
						("instrument_"++(n)++"_full").asSymbol.envirPut(("instrument_"++(n)++"_full").asSymbol.envirGet.flatten);
					});
				};



				" 3".postln;
				//create pseq arrays function

				(
					~create_pseq_arrays_func = {

						h = 0.midicps;

						if(~tuning_frequency != 440, {
							s = 440.cpsmidi - ~tuning_frequency.cpsmidi;
							if(s.isNegative == true, {s = s.neg});
							case
							{440.cpsmidi > ~tuning_frequency.cpsmidi}{~midi_shift = s.neg}
							{440.cpsmidi < ~tuning_frequency.cpsmidi}{~midi_shift = s};
						});

						1.do({
							/*
							~number_of_instruments.do(x = 0; {("instrument_"++(x=x+1)++"_full").asSymbol.envirPut([]);});*/

							/*
							("instrument_"++(1)++"_full").asSymbol.envirGet.at(5800).size
							*/

							case
							{~ai_assist_compose != 1}{
							if(~saved_piece_loaded != 1, {

								~number_of_instruments.do(x = 0; {
									x=x+1;
									/*~number_of_parts_in_piece.do(y = 0; {("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet.add(("instrument_"++(x)++"_array_part_"++(y=y+1)).asSymbol.envirGet););});
									*/("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet.flatten);
									("instrument_"++(x)++"_full_midi").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet);
									if(~tuning_frequency != 440, {("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet + ~midi_shift)});
									("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet.midicps);
									("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet.replace(h, 0));
								});
							}, {
								~number_of_instruments.do(x = 0; {
									x=x+1;
									("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full_midi").asSymbol.envirGet);
									if(~tuning_frequency != 440, {("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet + ~midi_shift)});
									("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet.midicps);
									("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet.replace(h, 0));
								});
								~saved_piece_loaded = 0;
							});
							}
							{~ai_assist_compose == 1}{
								nil
							};


							~number_of_instruments.do(x = 0; {
								x=x+1;
								2.do(s = 0; {
									s=s+1;
									("old_instrument_"++(x)++"_freq_"++(s)).asSymbol.envirPut(nil);
								});
							});

							"Piece Creation Is Finsihed".postln;

							if(~piece_created == nil, {~piece_created = "true";});
							AppClock.sched(0, {
								~playbutton_func.value;
								~piecegenerated.string = "Piece Generated";
								AppClock.sched(3, {~piecegenerated.close});
							});

							AppClock.sched(0, {
								~fullpiecedur1 = ((~piecedur_secs = (("instrument_"++(1)++"_full").asSymbol.envirGet.size*~bit_duration).round)/60);
								~dd_secs = ((~fullpiecedur1-(~dd_mins = ~fullpiecedur1.roundUp-1))*60).roundUp;
								/*~fullpiecedur = ("piece length - "++(~dd_mins.asString)++":"++(if(~dd_secs<10,{"0"},{""})++~dd_secs.asString)++" mins");
								*/
								~fullpiecedur = ("Piece Length - "++(("instrument_"++(1)++"_full").asSymbol.envirGet.size.asString)++" bits ("++(~dd_mins.asInteger.asString)++":"++(if(~dd_secs<10,{"0"},{""})++~dd_secs.asInteger.asString)++" mins)");
								~piecedurationstatictext.string = ~fullpiecedur;
								("\npiece duration: \n"++((~dd_mins.asInteger.asString)++":"++(if(~dd_secs<10,{"0"},{""})++~dd_secs.asInteger.asString))++" minutes";).postln;
								/*rangeslider.set(spec: ControlSpec(0, if(~piecedur_secs != nil, {~piecedur_secs}, {0}), \lin, 1, 50, "secs"), initVal: [0,if(~piecedur_secs != nil, {~piecedur_secs}, {0})]);
								*/~timeslider.set(spec: ControlSpec(0, ("instrument_"++(1)++"_full").asSymbol.envirGet.size, \lin, 1, 50, "bits"));
								~rate_selection_start = 0;
								~rate_selection_end = ("instrument_"++(1)++"_full").asSymbol.envirGet.size;
							});

							if(~quicknewpiecebutton_num == 1, {
								AppClock.sched(0, {
									~quicknewpiecebutton_num = 0;
									~stopbuttonfunc.value;
									~playbutton.value = 1;
									~run_piece_func.value;
								});
							});
						});
					};

					~create_pseq_arrays_func.value;
				);
			};

			~create_piece_function.value;

		);

		/*~create_piece_routine.isPlaying*/

	);
};