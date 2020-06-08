~number_of_instruments = 3;
g = File((~ddbot_root)++"/program_files/other_program_files/numberofvoices.scd", "w");
g.write(~number_of_instruments.asString);
g.close;
~num_assist_parts = 0;
~line_multiple1 = 1;
~line_multiple2 = 0;