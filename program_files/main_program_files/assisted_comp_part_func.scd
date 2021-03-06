~find_notes_func = {

	if(((e != nil) and: (e != [])), {

		~number_of_instruments.do(~inst_num = 0; {

			~inst_num = ~inst_num + 1;

			// find old notes of instrument

			("instrument_"++(~inst_num)++"_freq").asSymbol.envirPut(
				~find_notes_array.value.at(
					~find_notes_array.value.size - 1;
				);
			);

			u = 0;
			q = ~find_notes_array.value.size;
			(case
				{~number_of_instruments == 1}{2}
				{~number_of_instruments == 2}{1}
				{~number_of_instruments == 3}{1};).do({
				//find last notes
				c = ~find_notes_array.value.at((q-1));
				b = ~find_notes_array.value.at((q-2));
				if(((b != 0) and: (b != nil) and: (c != 0) and: (c != nil)), {
					while({c == b}, {
						q=q-1;
						c = ~find_notes_array.value.at((q-1));
						b = ~find_notes_array.value.at((q-2));
					});
					if(((b != 0) and: (b != nil)), {
						q=q-1;
						("old_instrument_"++(~inst_num)++"_freq_"++(u=u+1)).asSymbol.envirPut(b);
					});
				}, {
					("old_instrument_"++(~inst_num)++"_freq_"++(u=u+1)).asSymbol.envirPut(12);
				});
			});

		});

		~number_of_instruments.do(~inst_num = 0; {

			~inst_num = ~inst_num + 1;

			u = 0;
			q = ~find_notes_array.value.size;
			(case
				{~number_of_instruments == 1}{2}
				{~number_of_instruments == 2}{1}
				{~number_of_instruments == 3}{1};).do({

				k = 1;
				c = ~find_notes_array.value.at(~rate_selection_start+(q-1));
				b = ~find_notes_array.value.at(~rate_selection_start+(q-2));
				if(((b != 0) and: (b != nil) and: (c != 0) and: (c != nil)), {
					while({c == b}, {
						q=q-1;
						c = ~find_notes_array.value.at(~rate_selection_start+(q-1));
						b = ~find_notes_array.value.at(~rate_selection_start+(q-2));
						k = k + 1;
					});
					("length"++(u=u+1)).asSymbol.envirPut(k);
					q=q-1;
				}, {
					("length"++(u=u+1)).asSymbol.envirPut(
						if(("length"++(1)).asSymbol.envirGet != nil, {
							("length"++(u-1)).asSymbol.envirGet;
						}, {0});
					);
				});

				("bit_clock_"++(~inst_array.wrapAt(~inst_num-1))++"_choice").asSymbol.envirPut(
					("length"++(1)).asSymbol.envirGet;
				);

				("bit_clock_"++(~inst_array.wrapAt(~inst_num-1))).asSymbol.envirPut(
					("length"++(1)).asSymbol.envirGet;
				);

				if(("length"++(2)).asSymbol.envirGet != nil, {
					("old_bit_clock_"++(~inst_num)++"_choice_1").asSymbol.envirPut(
						("length"++(2)).asSymbol.envirGet;
					);
				});
			});

		});

	}, {
		~number_of_instruments.do(~inst_num = 0; {
			~inst_num = ~inst_num + 1;
			("instrument_"++(~inst_num)++"_freq").asSymbol.envirPut(rrand(~midi_low_limit, ~midi_high_limit));
		});

		~number_of_instruments.do(n = 0; {("bit_clock_"++(n=n+1)).asSymbol.envirPut(0);});
		~number_of_instruments.do(n = 0; {("bit_clock_"++(n=n+1)++"_choice").asSymbol.envirPut(~bit_choice_array.choose)});
		~number_of_instruments.do(n = 0; {("old_bit_clock_"++(n=n+1)++"_choice_1").asSymbol.envirPut(("bit_clock_"++(n)++"_choice").asSymbol.envirGet)});
		~number_of_instruments.do(x = 0; {("instrument_"++(x=x+1)++"_if_zero").asSymbol.envirPut("false");});


		//generation of piece

		~number_of_instruments.do(x = 0; {("instrument_"++(x=x+1)++"_if_zero").asSymbol.envirPut("false");});

		~number_of_instruments.do(x = 0; {
			x=x+1;
			("instrument_"++(x)++"_array").asSymbol.envirPut(
				("instrument_"++(x)++"_array").asSymbol.envirGet.add((("instrument_"++(x)++"_freq").asSymbol.envirPut(rrand(~midi_low_limit, ~midi_high_limit))).value)
			);
			("bit_clock_"++(x)++"_choice").asSymbol.envirPut(~bit_choice_array.choose);
		});

		~number_of_instruments.do(s = 0; {("old_instrument_"++(s=s+1)++"_freq_1").asSymbol.envirPut(("instrument_"++(s)++"_array").asSymbol.envirGet.at(0).asInteger + 12)});

	});

};

~last_notes_func = {

	c = ~c_assist_part_number;
	a = {("assist_inst_"++(1)++"_array_part_"++(c-1)++"_midi").asSymbol.envirGet;};
	e = a.value;
	while({(((e == nil) or: (e == [])) and: (c >= 2))}, {
		c = c - 1;
		e = a.value;

	});
	~e_last = e;
	~last_part_num = c-1;
	if(~gen_c_on != 1, {
		~find_notes_array = {("assist_inst_"++(~inst_num)++"_array_part_"++(~last_part_num)++"_midi").asSymbol.envirGet;};
		~find_notes_func.value;
		("length"++(1)).asSymbol.envirPut(nil);
		("length"++(2)).asSymbol.envirPut(nil);
	});
	~gen_c_on = 0;
};

~next_notes_func = {

	c = ~c_assist_part_number;
	a = {("assist_inst_"++(1)++"_array_part_"++(c+1)++"_midi").asSymbol.envirGet;};
	e = a.value;
	while({(((e == nil) or: (e == [])) and: (c < ~num_assist_parts))}, {
		c = c + 1;
		e = a.value;
	});
	~e_next = e;
	~next_part_num = c+1;
	if(~gen_c_on != 1, {
		~find_notes_array = {("assist_inst_"++(~inst_num)++"_array_part_"++(~next_part_num)++"_midi").asSymbol.envirGet.reverse;};
		~find_notes_func.value;
		("length"++(1)).asSymbol.envirPut(nil);
		("length"++(2)).asSymbol.envirPut(nil);
	});
	~gen_c_on = 0;
};

~assisted_comp_part_func = {

	~number_of_instruments.do(n = 0; {("bit_clock_"++(n=n+1)).asSymbol.envirPut(0);});

	x = 0;
	~inst_array = Array.fill(~number_of_instruments, {x=x+1});

	~randseed_1 =  99999999.rand;
	~randseed_2 =  99999999.rand;
	~randseed_3 =  99999999.rand;

	case
	{~composition_process_setting == 2}{

		thisThread.randSeed = ~randseed_1;

		~c_assist_part_number = ~assist_part_number;

		~last_notes_func.value;

		thisThread.randSeed = ~randseed_2;

		(~part_length_func.value).do({~inst_clock_func.value;});

		~number_of_instruments.do(n = 0; {
			n=n+1;
			("assist_inst_"++(n)++"_array_part_"++(~assist_part_number)++"_midi").asSymbol.envirPut(("instrument_"++(n)++"_array").asSymbol.envirGet;);
			("instrument_"++(n)++"_array").asSymbol.envirPut([]);
		});

	}
	{~composition_process_setting == 1}{

		// find last notes of last sequence

		~c_assist_part_number = ~assist_part_number;

		~last_notes_func.value;

		thisThread.randSeed = ~randseed_1;

		~number_of_instruments.do(n = 0; {
			n=n+1;
			("instrument_"++(n)++"_array").asSymbol.envirPut([]);
			("instrument_"++(n)++"_array_1").asSymbol.envirPut([]);
		});
		(~part_length_func.value).do({~inst_clock_func.value});
		~number_of_instruments.do(n = 0; {("instrument_"++(n=n+1)++"_array_1").asSymbol.envirPut(("instrument_"++(n)++"_array").asSymbol.envirGet)});

		// same with first notes

		~next_notes_func.value;

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

		if(~last_part_num > 0, {
			~number_of_instruments.do(n = 0; {
				n=n+1;
				("note_1_inst_"++(n)++"_part_"++(~assist_part_number)).asSymbol.envirPut(
					("note_1_inst_"++(n)++"_part_"++(~last_part_num)).asSymbol.envirGet;
				);

				("note_2_inst_"++(n)++"_part_"++(~assist_part_number)).asSymbol.envirPut(
					("note_2_inst_"++(n)++"_part_"++(~last_part_num)).asSymbol.envirGet;
				);

				("comp_int_2_inst_"++(n)++"part_"++((~assist_part_number))).asSymbol.envirPut(
					if(("comp_int_2_inst_"++(n)++"part_"++(~last_part_num)).asSymbol.envirGet != nil, {
						("comp_int_2_inst_"++(n)++"part_"++(~last_part_num)).asSymbol.envirGet;
					}, {
						("assist_inst_"++(n)++"_comp_interval_array_part_"++(~last_part_num)).asSymbol.envirGet.at(
							(("assist_inst_"++(n)++"_comp_interval_array_part_"++(~last_part_num)).asSymbol.envirGet.size-1).round;
						);
					});/*{("assist_inst_"++(n)++"_comp_interval_array_part_"++(~last_part_num)).asSymbol.envirGet.at(
					(("assist_inst_"++(n)++"_comp_interval_array_part_"++(~last_part_num)).asSymbol.envirGet.size-1).round;
					);
					}*/
				);
			});
		}, {
			~number_of_instruments.do(n = 0; {
				n=n+1;
				("note_1_inst_"++(n)++"_part_"++(~assist_part_number)).asSymbol.envirPut(nil);
				("note_2_inst_"++(n)++"_part_"++(~assist_part_number)).asSymbol.envirPut(nil);
				("comp_int_2_inst_"++(n)++"part_"++((~assist_part_number))).asSymbol.envirPut(0);
			});
		});
		~number_of_instruments.do(n = 0; {("assist_inst_"++(n=n+1)++"_comp_interval_array_part_"++(~assist_part_number)).asSymbol.envirPut([]);});
		~number_of_instruments.do(n = 0; {("assist_inst_"++(n=n+1)++"_comp_choice_array_part_"++(~assist_part_number)).asSymbol.envirPut([]);});

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
							(("note_1_inst_"++(n)++"_part_"++(~assist_part_number)).asSymbol.envirGet - ("note_2_inst_"++(n)++"_part_"++(~assist_part_number)).asSymbol.envirGet).round;

						);

						if(("comp_int_1_inst_"++(n)).asSymbol.envirGet.isNegative == true, {
							("comp_int_1_inst_"++(n)).asSymbol.envirPut(
								("comp_int_1_inst_"++(n)).asSymbol.envirGet.neg

							)
						});

						while({("comp_int_1_inst_"++(n)).asSymbol.envirGet > 11}, {
							("comp_int_1_inst_"++(n)).asSymbol.envirPut(
								("comp_int_1_inst_"++(n)).asSymbol.envirGet - 12

							)
						});

						~comp_choice = ~blend_choice_array.wchoose(
							~ml_data_array.at(
								("comp_int_1_inst_"++(n)).asSymbol.envirGet*(~blend_choice_array.size-1)+("comp_int_2_inst_"++(n)++"part_"++((~assist_part_number))).asSymbol.envirGet.round;

							).normalizeSum;

						);

						("comp_result_"++(n)).asSymbol.envirPut(
							("comp_result_"++(n)).asSymbol.envirGet + ~comp_choice.value

						);

						if(((("comp_result_"++(n)).asSymbol.envirGet > ~midi_high_limit) or: (("comp_result_"++(n)).asSymbol.envirGet < ~midi_low_limit)), {
							while({("comp_result_"++(n)).asSymbol.envirGet > ~midi_high_limit}, {("comp_result_"++(n)).asSymbol.envirPut(("comp_result_"++(n)).asSymbol.envirGet-12)});
							while({("comp_result_"++(n)).asSymbol.envirGet < ~midi_low_limit}, {("comp_result_"++(n)).asSymbol.envirPut(("comp_result_"++(n)).asSymbol.envirGet+12)});
						});

						("comp_int_2_inst_"++(n)++"part_"++((~assist_part_number))).asSymbol.envirPut(
							("comp_int_1_inst_"++(n)).asSymbol.envirGet

						);
					});

					("instrument_"++(n)++"_array_blended").asSymbol.envirPut(("instrument_"++(n)++"_array_blended").asSymbol.envirGet.add(("comp_result_"++(n)).asSymbol.envirGet;));

					("assist_inst_"++(n)++"_comp_interval_array_part_"++(~assist_part_number)).asSymbol.envirPut(("assist_inst_"++(n)++"_comp_interval_array_part_"++(~assist_part_number)).asSymbol.envirGet.add(("comp_int_1_inst_"++(n)).asSymbol.envirGet.round));

					("assist_inst_"++(n)++"_comp_choice_array_part_"++(~assist_part_number)).asSymbol.envirPut(("assist_inst_"++(n)++"_comp_choice_array_part_"++(~assist_part_number)).asSymbol.envirGet.add(~comp_choice.value));

					("assist_inst_"++(n)++"_array_part_"++(~assist_part_number)++"_midi").asSymbol.envirPut(("instrument_"++(n)++"_array_blended").asSymbol.envirGet;);

				}, {
					u = ((("instrument_"++(n)++"_array_1").asSymbol.envirGet.at(i) + ("instrument_"++(n)++"_array_2").asSymbol.envirGet.at(i)) / 2).round;

					("instrument_"++(n)++"_array_blended").asSymbol.envirPut(("instrument_"++(n)++"_array_blended").asSymbol.envirGet.add(u.value));

					("assist_inst_"++(n)++"_array_part_"++(~assist_part_number)++"_midi").asSymbol.envirPut(("instrument_"++(n)++"_array_blended").asSymbol.envirGet;);

				});
			});
		});

	};

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

		~number_of_instruments.do(x = 0; {
			x=x+1;
			("assist_inst_"++(n)++"_array_part_"++(~assist_part_number)++"midi_backup").asSymbol.envirPut(("assist_inst_"++(n)++"_array_part_"++(~assist_part_number)++"_midi").asSymbol.envirGet;);
			/*~number_of_parts_in_piece.do(y = 0; {("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet.add(("instrument_"++(x)++"_array_part_"++(y=y+1)).asSymbol.envirGet););});
			*/("assist_inst_"++(x)++"_array_part_"++(~assist_part_number)++"_midi").asSymbol.envirPut(("assist_inst_"++(x)++"_array_part_"++(~assist_part_number)++"_midi").asSymbol.envirGet.flatten);
			/*("instrument_"++(x)++"_full_midi").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet);*/
			if(~tuning_frequency != 440, {("assist_inst_"++(x)++"_array_part_"++(~assist_part_number)++"_midi").asSymbol.envirPut(("assist_inst_"++(x)++"_array_part_"++(~assist_part_number)++"_midi").asSymbol.envirGet + ~midi_shift)});
			("assist_inst_"++(x)++"_array_part_"++(~assist_part_number)).asSymbol.envirPut(("assist_inst_"++(x)++"_array_part_"++(~assist_part_number)++"_midi").asSymbol.envirGet.midicps);
			("assist_inst_"++(x)++"_array_part_"++(~assist_part_number)).asSymbol.envirPut(("assist_inst_"++(x)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet.replace(h, 0));
		});
	});

};