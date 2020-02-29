// BUILD/GENERATE/COMPOSE PIECE

~clock_diff_0_func = {
	~clock_diff_0 = ("instrument_"++(~inst_array.wrapAt(~inst_num-1))++"_freq").asSymbol.envirGet - ("old_instrument_"++(~inst_array.wrapAt(~inst_num-1))++"_freq").asSymbol.envirGet;
	if(~clock_diff_0.isFloat == true, {~clock_diff_0 = ~clock_diff_0.round});
	if(~clock_diff_0.isNegative == true, {~clock_diff_0 = ~clock_diff_0.neg});
	while({~clock_diff_0 > 12}, {~clock_diff_0 = ~clock_diff_0 - 12});
};

~clock_diff_1_func = {
	~clock_diff_1 = ("instrument_"++(~inst_array.wrapAt(~inst_num-1))++"_freq").asSymbol.envirGet - ("instrument_"++(~inst_array.wrapAt(~inst_num))++"_freq").asSymbol.envirGet;
	if(~clock_diff_1.isFloat == true, {~clock_diff_1 = ~clock_diff_1.round});
	if(~clock_diff_1.isNegative == true, {~clock_diff_1 = ~clock_diff_1.neg});
	while({~clock_diff_1 > 12}, {~clock_diff_1 = ~clock_diff_1 - 12});
};

~clock_diff_2_func = {
	~clock_diff_2 = ("instrument_"++(~inst_array.wrapAt(~inst_num-1))++"_freq").asSymbol.envirGet - ("instrument_"++(~inst_array.wrapAt(~inst_num+1))++"_freq").asSymbol.envirGet;
	if(~clock_diff_2.isFloat == true, {~clock_diff_2 = ~clock_diff_2.round});
	if(~clock_diff_2.isNegative == true, {~clock_diff_2 = ~clock_diff_2.neg});
	while({~clock_diff_2 > 12}, {~clock_diff_2 = ~clock_diff_2 - 12});
};

~note_generator_func = {

	case
	{~number_of_instruments == 1}{1.do(s = (-1); {("clock_diff_"++(s=s+1)++"_func").asSymbol.envirGet.value;});}
	{~number_of_instruments == 2}{2.do(s = (-1); {("clock_diff_"++(s=s+1)++"_func").asSymbol.envirGet.value;});}
	{~number_of_instruments == 3}{3.do(s = (-1); {("clock_diff_"++(s=s+1)++"_func").asSymbol.envirGet.value;});};

	(~number_of_instruments-1).do(u = (-1); {
		u=u+1;
		("clock_diff_"++(u)).asSymbol.envirPut(("clock_diff_"++(u)).asSymbol.envirGet.round.asInteger;);
	});

	~intv_weights = FileReader.read((~ddbot_root)++"/program_files/other_program_files/ddbot_training_data_folder/ddbot_training_data/ddbot_training_data_current_"++(~number_of_instruments.asString)++"_inst/ddbot_interval_weights/"++(~player_array.wrapAt(~inst_num-1))++"_interval_weights/"++(
		case
		{~number_of_instruments == 1}{(~clock_diff_0.asString)}
		{~number_of_instruments == 2}{(~clock_diff_0.asString)++"__"++(~clock_diff_1.asString)}
		{~number_of_instruments == 3}{(~clock_diff_0.asString)++"__"++(~clock_diff_1.asString)++"_"++(~clock_diff_2.asString)};
	)++"_weights.scd", false, true);
	~intv_weights = ~intv_weights.flatten.asFloat;
	~intv_weights = ~intv_weights.normalizeSum;
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
		{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(0)) and: (("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(1))}{if((("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(0)+~time_array.at(1)/2)), {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(0)}, {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(1)})}
		{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(1)) and: (("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(2))}{if((("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(1)+~time_array.at(2)/2)), {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(1)}, {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(2)})}
		{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(2)) and: (("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(3))}{if((("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(2)+~time_array.at(3)/2)), {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(2)}, {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(3)})}
		{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(3)) and: (("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(4))}{if((("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(3)+~time_array.at(4)/2)), {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(3)}, {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(4)})}
		{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(4)) and: (("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(5))}{if((("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(4)+~time_array.at(5)/2)), {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(4)}, {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(5)})}
		{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(5)) and: (("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(6))}{if((("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(5)+~time_array.at(6)/2)), {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(5)}, {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(6)})}
		{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(6)) and: (("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(7))}{if((("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(6)+~time_array.at(7)/2)), {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(6)}, {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(7)})}
		{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(7))}{("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(7)};
	});


	/*
	("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+2))++"_choice").asSymbol.envirGet
	("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+2))++"_place").asSymbol.envirGet
	*/

	~time_weights = FileReader.read((~ddbot_root)++"/program_files/other_program_files/ddbot_training_data_folder/ddbot_training_data/ddbot_training_data_current_"++(~number_of_instruments.asString)++"_inst/ddbot_time_weights/"++(~player_array.wrapAt(~inst_num-1))++"_time_weights/"++(
		case
		{~number_of_instruments == 1}{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1))++"_place").asSymbol.envirGet.asString)}
		{~number_of_instruments == 2}{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1))++"_place").asSymbol.envirGet.asString)++"_"++(("dna_clock_"++(~inst_array.wrapAt(~inst_num))++"_place").asSymbol.envirGet.asString)}
		{~number_of_instruments == 3}{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1))++"_place").asSymbol.envirGet.asString)++"_"++(("dna_clock_"++(~inst_array.wrapAt(~inst_num))++"_place").asSymbol.envirGet.asString)++"_"++(("dna_clock_"++(~inst_array.wrapAt(~inst_num+1))++"_place").asSymbol.envirGet.asString)};
	)++"_time_weights.scd", false, true);
	~time_weights = ~time_weights.flatten.asFloat;
	~time_weights = ~time_weights.normalizeSum;
	~new_time = ~dna_choice_array.wchoose(~time_weights);

	("dna_clock_"++(~inst_num)++"_choice").asSymbol.envirPut(~new_time);
};

~inst_clock_func = {

	~number_of_instruments.do(n = 0; {("dna_clock_"++(n=n+1)).asSymbol.envirPut(("dna_clock_"++(n)).asSymbol.envirGet+1);});
	~number_of_instruments.do(n = 0; {
		n=n+1;
		~inst_num = n;
		("instrument_"++(~inst_num)++"_array").asSymbol.envirPut(
			("instrument_"++(~inst_num)++"_array").asSymbol.envirGet.add(
				if(("dna_clock_"++(~inst_num)).asSymbol.envirGet == ("dna_clock_"++(~inst_num)++"_choice").asSymbol.envirGet, {
					("dna_clock_"++(~inst_num)).asSymbol.envirPut((-1));
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


~midi_low_limit = 45;
~midi_high_limit = 81;
~tuning_frequency = 440; //default 440 hz
~freqmap = (i = ~tuning_frequency / (2 ** (69 / 12)); a = (-1); b = (-1); Array.fill(128, {i * (2 ** (((a=a+1) / 12).trunc)) * ( (2 ** (12.reciprocal)) ** ((b=b+1)  % 12));}));/*~freqmap = (i = 8.0271480262684/*-0.0000000000024*/; ~stepratio = 1.0594630943593; Array.fill(128, { i = i*~stepratio;}));*/
~freqmap = ~freqmap.cpsmidi;
/*~freqmap2 = (i = ~freqmap.at(58)/*-0.0000000000024*/; ~stepratio = 1.0594630943593; Array.fill(24, { i = i*~stepratio;}));*/
~freqmap2 = (n = ~midi_low_limit-1; Array.fill(~midi_high_limit-~midi_low_limit+1, {~freqmap.at(n=n+1);}));
/*~ratio_array = (x = 0; Array.fill(12, {1.0594630943593**(x=x+1)}));*/
~interval_array = (1..12);
~midi_ratio_array = (1..12);


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

			/*~dna_choice_array = (1..64);*/
			~time_array = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8];
			~dna_choice_array = [1, 2, 4, 8, 16, 32, 64, 128];
			~dna_choice_array_num = (1..~dna_choice_array.size);

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

				~number_of_instruments.do(x = 0; {
					x=x+1;
					("instrument_"++(x)++"_full").asSymbol.envirPut([]);
					("instrument_"++(x)++"_array").asSymbol.envirPut([]);
				});

				// choose compoition process

				~composition_process_setting = 1;

				case
				{~composition_process_setting == 1}{

					// I NEED DDBOT TO SOUND *KINDA ROBOTIC, AND KINDA HUMAN*


					// Top level -
					// Generate one random chord. Having nothing to do with time, generate follow up chords in the sequence using ML data.
					// Level 3 -
					// For each chord in above level sequence, do same thing as top level.
					// Level 2 -
					// " "
					// Level 1 (when ~create_piece_function fires) -
					// For each chord

					// For now... override

					// Have DDbot machine learn preferences in that^


					// generate top-level chords

					~number_of_instruments.do(x = 0; {
						x=x+1;
						("top_level_notes_"++(x)).asSymbol.envirPut([]);
					});

					thisThread.randSeed = ~randseed_2;

					//generate first set of notes

					~number_of_instruments.do(n = 0; {
						~inst_num = (n=n+1);
						("top_level_notes_"++(~inst_array.wrapAt(~inst_num-1))).asSymbol.envirPut([rrand(~midi_low_limit, ~midi_high_limit)]);
						("instrument_"++(~inst_array.wrapAt(~inst_num-1))++"_freq").asSymbol.envirPut(("top_level_notes_"++(~inst_array.wrapAt(~inst_num-1))).asSymbol.envirGet.at(0).asInteger);
					});

					~number_of_instruments.do(s = 0; {("old_instrument_"++(s=s+1)++"_freq").asSymbol.envirPut(("top_level_notes_"++(s)).asSymbol.envirGet.at(0).asInteger + 12)});

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

					~number_of_instruments.do(n = 0; {("dna_clock_"++(n=n+1)).asSymbol.envirPut((-1));});
					~number_of_instruments.do(n = 0; {("dna_clock_"++(n=n+1)++"_choice").asSymbol.envirPut(~dna_choice_array.choose)});
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
						(o = (exprand(4, 16).round)/~bit_duration).do({~inst_clock_func.value});
						~number_of_instruments.do(n = 0; {("instrument_"++(n=n+1)++"_array_1").asSymbol.envirPut(("instrument_"++(n)++"_array").asSymbol.envirGet)});

					};

					~composition_setting = 2;

					//compose action

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

						// reverse array 2, then blend
						~number_of_instruments.do(n = 0; {("instrument_"++(n=n+1)++"_array_2_forwards").asSymbol.envirPut(("instrument_"++(n)++"_array_2").asSymbol.envirGet)});
						~number_of_instruments.do(n = 0; {("instrument_"++(n=n+1)++"_array_2").asSymbol.envirPut(("instrument_"++(n)++"_array_2").asSymbol.envirGet.reverse)});

						~number_of_instruments.do(n = 0; {n=n+1; ("instrument_"++(n)++"_array_blended").asSymbol.envirPut([]);});

						~number_of_instruments.do(n = 0; {
							n=n+1;
							o.do(i = (-1); {
								i=i+1;
								u = ((("instrument_"++(n)++"_array_1").asSymbol.envirGet.at(i) + ("instrument_"++(n)++"_array_2").asSymbol.envirGet.at(i)) / 2).round;
								("instrument_"++(n)++"_array_blended").asSymbol.envirPut(("instrument_"++(n)++"_array_blended").asSymbol.envirGet.add(u.value));
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
				{~composition_process_setting == 2}{

					thisThread.randSeed = ~randseed_2;

					~part_bit_size = 64;
					~max_number_of_parts = 200; //max 200 parts. random probability choice decreases linearly from "~max_number_of_parts" value to 1.
					~number_of_parts_in_piece = (~max_number_of_parts.linrand - ~max_number_of_parts).neg;
					~just_ml_piece_size = ~part_bit_size*~number_of_parts_in_piece;

					thisThread.randSeed = ~randseed_3;

					~number_of_instruments.do(y = 0; {("dna_clock_"++(y=y+1)).asSymbol.envirPut(0);});


					//generation of piece

					~number_of_instruments.do(x = 0; {("instrument_"++(x=x+1)++"_if_zero").asSymbol.envirPut("false");});

					~number_of_instruments.do(x = 0; {
						x=x+1;
						("instrument_"++(x)++"_array").asSymbol.envirPut(
							("instrument_"++(x)++"_array").asSymbol.envirGet.add((("instrument_"++(x)++"_freq").asSymbol.envirPut(~freqmap2.choose)).value)
						);
						("dna_clock_"++(x)++"_choice").asSymbol.envirPut(~dna_choice_array.choose);
					});

					~number_of_instruments.do(s = 0; {("old_instrument_"++(s=s+1)++"_freq").asSymbol.envirPut(("instrument_"++(s)++"_array").asSymbol.envirGet.at(0).asInteger + 12)});

					/*
					("dna_clock_"++(3)++"_choice").asSymbol.envirGet*/

					/*(~number_of_instruments-1).do(x = 1; {
					("instrument_"++(x=x+1)++"_freq").asSymbol.envirPut(0);
					("dna_clock_"++(x)++"_choice").asSymbol.envirPut(0);/*
					("dna_clock_"++(2)++"_choice").asSymbol.envirGet <~time_array.at(0)           (0);*/
					});*/

					/*(~number_of_instruments-1).do(s = 0; {("dna_clock_"++(s=s+1)).asSymbol.envirPut(("dna_clock_"++(s)).asSymbol.envirGet+1);});
					(~number_of_instruments-1).do(s = 1; {
					s=s+1;
					~dna_clock = ("dna_clock_"++(s)).asSymbol.envirGet;
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

						1.do({
							/*
							~number_of_instruments.do(x = 0; {("instrument_"++(x=x+1)++"_full").asSymbol.envirPut([]);});*/

							/*
							("instrument_"++(1)++"_full").asSymbol.envirGet.at(5800).size
							*/

							if(~saved_piece_loaded != 1, {

								~number_of_instruments.do(x = 0; {
									x=x+1;
									/*~number_of_parts_in_piece.do(y = 0; {("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet.add(("instrument_"++(x)++"_array_part_"++(y=y+1)).asSymbol.envirGet););});
									*/("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet.flatten);
									("instrument_"++(x)++"_full_midi").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet);
									("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet.midicps);
									("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet.replace(h, 0));
								});
							}, {
								~number_of_instruments.do(x = 0; {
									x=x+1;
									("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full_midi").asSymbol.envirGet.midicps);
									("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet.replace(h, 0));
								});
								~saved_piece_loaded = 0;
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

							if(~quicknewpiecebutton == 1, {
								AppClock.sched(0, {
									~quicknewpiecebutton = 0;
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