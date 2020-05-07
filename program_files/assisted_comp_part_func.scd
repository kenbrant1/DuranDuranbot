~assisted_comp_part_func = {

	~randseed_1 =  99999999.rand;
	~randseed_2 =  99999999.rand;
	~randseed_3 =  99999999.rand;

	thisThread.randSeed = ~randseed_1;

	~number_of_instruments.do(n = 0; {("bit_clock_"++(n=n+1)).asSymbol.envirPut((-1));});
	~number_of_instruments.do(n = 0; {("bit_clock_"++(n=n+1)++"_choice").asSymbol.envirPut(~bit_choice_array.choose)});
	~number_of_instruments.do(n = 0; {("old_bit_clock_"++(n=n+1)++"_choice_1").asSymbol.envirPut(("bit_clock_"++(n)++"_choice").asSymbol.envirGet)});
	~number_of_instruments.do(x = 0; {("instrument_"++(x=x+1)++"_if_zero").asSymbol.envirPut("false");});
	/*~number_of_instruments.do(n = 0; {("bottom_level_notes_"++(n=n+1)).asSymbol.envirPut([])});*/

	// find last notes of last sequence

	~number_of_instruments.do(~inst_num = 0; {
		~inst_num = ~inst_num + 1;
		c = ~assist_part_number;
		e = {("assist_inst_"++(~inst_num)++"_freq_part_"++(c-1)).asSymbol.envirGet;};
		e = e.value;
		while({((e == nil) and: (c > 0))}, {
			c = c - 1;
			e = e.value;

		});
		("last_part_num_inst_"++(~inst_num)++"_last_note").asSymbol.envirPut(c);
		if(e != nil, {
			("assist_inst_"++(~inst_num)++"_last_note").asSymbol.envirPut(
				("assist_inst_"++(~inst_num)++"_freq_part_"++(c-1)).asSymbol.envirGet.at(
					("assist_inst_"++(~inst_num)++"_freq_part_"++(c-1)).asSymbol.envirGet.size - 1;
				);
			);
		}, {
			("assist_inst_"++(~inst_num)++"_last_note").asSymbol.envirPut(rrand(~midi_low_limit, ~midi_high_limit));
		});

	});

	// find first notes of next sequence

	~number_of_instruments.do(~inst_num = 0; {
		~inst_num = ~inst_num + 1;
		c = ~assist_part_number;
		e = {("assist_inst_"++(~inst_num)++"_freq_part_"++(c+1)).asSymbol.envirGet;};
		e = e.value;
		while({((e == nil) and: (c < ~num_assist_parts))}, {
			c = c + 1;
			e = e.value;

		});
		if(e != nil, {
			("assist_inst_"++(~inst_num)++"_first_note").asSymbol.envirPut(
				("assist_inst_"++(~inst_num)++"_freq_part_"++(c+1)).asSymbol.envirGet.at(0);
			);
		}, {
			("assist_inst_"++(~inst_num)++"_first_note").asSymbol.envirPut(rrand(~midi_low_limit, ~midi_high_limit));
		});

	});

	// "put" last notes into ~instrument_n_freq's... and generate sequence

	~number_of_instruments.do(~inst_num = 0; {
		~inst_num = ~inst_num + 1;
		("instrument_"++(~inst_num)++"_freq").asSymbol.envirPut(("assist_inst_"++(~inst_num)++"_last_note").asSymbol.envirGet;);
	});

	thisThread.randSeed = ~randseed_2;

	~number_of_instruments.do(n = 0; {
		n=n+1;
		("instrument_"++(n)++"_array").asSymbol.envirPut([]);
		("instrument_"++(n)++"_array_1").asSymbol.envirPut([]);
	});
	(o = (exprand(4, 16).round)/~bit_duration).do({~inst_clock_func.value});
	~number_of_instruments.do(n = 0; {("instrument_"++(n=n+1)++"_array_1").asSymbol.envirPut(("instrument_"++(n)++"_array").asSymbol.envirGet)});

	// same with first notes

	~number_of_instruments.do(~inst_num = 0; {
		~inst_num = ~inst_num + 1;
		("instrument_"++(~inst_num)++"_freq").asSymbol.envirPut(("assist_inst_"++(~inst_num)++"_first_note").asSymbol.envirGet;);
	});

	~number_of_instruments.do(n = 0; {
		n=n+1;
		("instrument_"++(n)++"_array").asSymbol.envirPut([]);
		("instrument_"++(n)++"_array_2").asSymbol.envirPut([]);
	});
	o.do({~inst_clock_func.value});
	~number_of_instruments.do(n = 0; {("instrument_"++(n=n+1)++"_array_2").asSymbol.envirPut(("instrument_"++(n)++"_array").asSymbol.envirGet)});

	/*("assist_inst_"++(2)++"_first_note").asSymbol.envirGet*/

	/*~instrument_1_array_1
	~instrument_1_array_2
	~instrument_2_array_1
	~instrument_2_array_2*/

	// reverse array 2
	~number_of_instruments.do(n = 0; {("instrument_"++(n=n+1)++"_array_2").asSymbol.envirPut(("instrument_"++(n)++"_array_2").asSymbol.envirGet.reverse)});

	// blend the two arrays

	~number_of_instruments.do(n = 0; {n=n+1; ("instrument_"++(n)++"_array_blended").asSymbol.envirPut([]);});

	thisThread.randSeed = ~randseed_3;

	~number_of_instruments.do(n = 0; {
		n=n+1;
		if(("last_part_num_inst_"++(n)++"_last_note").asSymbol.envirGet > 0, {

			("note_1_inst_"++(n)++"_part_"++(~assist_part_number)).asSymbol.envirPut(
				("note_1_inst_"++(n)++"_part_"++(~last_part_num)).asSymbol.envirGet;
			);

			("note_2_inst_"++(n)++"_part_"++(~assist_part_number)).asSymbol.envirPut(
				("note_2_inst_"++(n)++"_part_"++(~last_part_num)).asSymbol.envirGet;
			);

			("comp_int_2_inst_"++(n)++"part_"++((~assist_part_number))).asSymbol.envirPut(
				("comp_int_2_inst_"++(n)++"part_"++(~last_part_num)).asSymbol.envirGet;
			);
		}, {
			("note_1_inst_"++(n)++"_part_"++(~assist_part_number)).asSymbol.envirPut(nil);
			("note_2_inst_"++(n)++"_part_"++(~assist_part_number)).asSymbol.envirPut(nil);
			("comp_int_2_inst_"++(n)++"part_"++((~assist_part_number))).asSymbol.envirPut(0);
		});
	});
	~number_of_instruments.do(n = 0; {("assist_instrument_"++(n=n+1)++"comp_interval_array_part"++(~assist_part_number)).asSymbol.envirPut([]);});
	~number_of_instruments.do(n = 0; {("assist_instrument_"++(n=n+1)++"comp_choice_array_part"++(~assist_part_number)).asSymbol.envirPut([]);});

	~number_of_instruments.do(n = 0; {var comp_choice;
		n=n+1;
		o.do(i = (-1); {
			i=i+1;
			if(~comp_machine_learning == 1, {

				if(((("instrument_"++(n)++"_array_1").asSymbol.envirGet.at(i) != ("note_1_inst_"++(n)++"_part_"++(~assist_part_number)).asSymbol.envirGet) or: (("instrument_"++(n)++"_array_2").asSymbol.envirGet.at(i) != ("note_2_inst_"++(n)++"_part_"++(~assist_part_number)).asSymbol.envirGet)), {

					("comp_result_"++(n)).asSymbol.envirPut(
						(((("note_1_inst_"++(n)++"_part_"++(~assist_part_number)).asSymbol.envirPut(
							("instrument_"++(n)++"_array_1").asSymbol.envirGet.at(i)
						).value

						)
						+ (("note_2_inst_"++(n)++"_part_"++(~assist_part_number)).asSymbol.envirPut(
							("instrument_"++(n)++"_array_2").asSymbol.envirGet.at(i)
						).value)) / 2).round;

					);

					("comp_int_1_inst_"++(n)).asSymbol.envirPut(
						("note_1_inst_"++(n)++"_part_"++(~assist_part_number)).asSymbol.envirGet - ("note_2_inst_"++(n)++"_part_"++(~assist_part_number)).asSymbol.envirGet

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
							("comp_int_1_inst_"++(n)).asSymbol.envirGet.value*(~blend_choice_array.size-1)+("comp_int_2_inst_"++(n)++"part_"++((~assist_part_number))).asSymbol.envirGet.value;

						).normalizeSum;

					);

					("comp_result_"++(n)).asSymbol.envirPut(
						("comp_result_"++(n)).asSymbol.envirGet + ~comp_choice.value

					);

					("comp_int_2_inst_"++(n)++"part_"++((~assist_part_number))).asSymbol.envirPut(
						("comp_int_1_inst_"++(n)).asSymbol.envirGet

					);
				});

				("instrument_"++(n)++"_array_blended").asSymbol.envirPut(("instrument_"++(n)++"_array_blended").asSymbol.envirGet.add(("comp_result_"++(n)).asSymbol.envirGet.value));

				("assist_instrument_"++(n)++"comp_interval_array_part"++(~assist_part_number)).asSymbol.envirPut(("assist_instrument_"++(n)++"comp_interval_array_part"++(~assist_part_number)).asSymbol.envirGet.add(("comp_int_1_inst_"++(n)).asSymbol.envirGet));

				("assist_instrument_"++(n)++"comp_choice_array_part"++(~assist_part_number)).asSymbol.envirPut(("assist_instrument_"++(n)++"comp_choice_array_part"++(~assist_part_number)).asSymbol.envirGet.add(~comp_choice.value));

				("assisted_inst_"++(n)++"_array_part_"++(~assist_part_number)++"_backup").asSymbol.envirPut(("assisted_inst_"++(n)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet;);
				("assisted_inst_"++(n)++"_array_part_"++(~assist_part_number)).asSymbol.envirPut(("instrument_"++(n)++"_array_blended").asSymbol.envirGet;);

			}, {
				u = ((("instrument_"++(n)++"_array_1").asSymbol.envirGet.at(i) + ("instrument_"++(n)++"_array_2").asSymbol.envirGet.at(i)) / 2).round;

				("instrument_"++(n)++"_array_blended").asSymbol.envirPut(("instrument_"++(n)++"_array_blended").asSymbol.envirGet.add(u.value));

			});
		});
	});



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

		if(~saved_piece_loaded != 1, {

			~number_of_instruments.do(x = 0; {
				x=x+1;
				/*~number_of_parts_in_piece.do(y = 0; {("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet.add(("instrument_"++(x)++"_array_part_"++(y=y+1)).asSymbol.envirGet););});
				*/("assisted_inst_"++(x)++"_array_part_"++(~assist_part_number)).asSymbol.envirPut(("assisted_inst_"++(x)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet.flatten);
				/*("instrument_"++(x)++"_full_midi").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet);*/
				if(~tuning_frequency != 440, {("assisted_inst_"++(x)++"_array_part_"++(~assist_part_number)).asSymbol.envirPut(("assisted_inst_"++(x)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet + ~midi_shift)});
				("assisted_inst_"++(x)++"_array_part_"++(~assist_part_number)).asSymbol.envirPut(("assisted_inst_"++(x)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet.midicps);
				("assisted_inst_"++(x)++"_array_part_"++(~assist_part_number)).asSymbol.envirPut(("assisted_inst_"++(x)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet.replace(h, 0));
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
	});

};