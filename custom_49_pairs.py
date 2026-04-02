from __future__ import annotations

PAIRED_STATEMENTS: list[tuple[str, str, str]] = [
    (
        "healthcare",
        "Healthcare is a fundamental human right. The government should provide universal healthcare coverage for all citizens, funded through progressive taxation. A single-payer system would reduce costs and eliminate the profit motive from medicine.",
        "Healthcare should be driven by free market competition. Government-run healthcare leads to inefficiency, long wait times, and reduced quality. Individuals should have the freedom to choose their own insurance plans and doctors.",
    ),
    (
        "gun_control",
        "We need stricter gun control laws including universal background checks, assault weapons bans, and red flag laws. The epidemic of gun violence in America demands urgent legislative action to protect public safety.",
        "The Second Amendment guarantees the individual right to bear arms. Gun control laws infringe on constitutional rights and only disarm law-abiding citizens. More guns in the hands of responsible citizens actually reduce crime.",
    ),
    (
        "immigration",
        "Immigrants strengthen our economy and enrich our culture. We should create pathways to citizenship for undocumented immigrants, protect DACA recipients, and reform our immigration system to be more welcoming and humane.",
        "We must secure our borders and enforce existing immigration laws. Illegal immigration depresses wages for American workers and strains public services. A strong border wall and strict enforcement are essential for national sovereignty.",
    ),
    (
        "climate",
        "Climate change is an existential crisis requiring immediate government action. We must transition to renewable energy, rejoin international climate agreements, and implement a Green New Deal to create jobs while saving the planet.",
        "Climate regulations destroy jobs and hurt the economy. The free market, not government mandates, should drive energy innovation. America should prioritize energy independence through domestic oil, gas, and clean coal production.",
    ),
    (
        "abortion",
        "Women have the fundamental right to make their own reproductive choices. Access to safe and legal abortion is essential healthcare. The government should not interfere with private medical decisions between a woman and her doctor.",
        "Life begins at conception and every unborn child deserves legal protection. Abortion is morally wrong and should be restricted. The government has a duty to protect the most vulnerable members of society, including the unborn.",
    ),
    (
        "taxation",
        "The wealthy and corporations should pay their fair share in taxes. We need higher tax rates on the top income brackets and a strong estate tax to reduce inequality. Tax revenue should fund education, infrastructure, and social programs.",
        "Lower taxes stimulate economic growth and job creation. Tax cuts allow businesses to invest and expand, benefiting everyone. The government should reduce spending rather than raise taxes. A flat tax would be the fairest system.",
    ),
    (
        "minimum_wage",
        "The federal minimum wage should be raised to at least fifteen dollars per hour. No one who works full-time should live in poverty. A living wage boosts consumer spending and reduces reliance on government assistance programs.",
        "Raising the minimum wage kills jobs, especially for small businesses and entry-level workers. Wages should be set by the free market based on supply and demand. Government-mandated wage floors lead to automation and unemployment.",
    ),
    (
        "criminal_justice",
        "Our criminal justice system is plagued by systemic racism and mass incarceration. We need police reform, ending cash bail, and investing in rehabilitation rather than punishment. Communities need social workers, not more police.",
        "Law and order is the foundation of a safe society. We need to support our police officers and give them the resources they need. Criminals should face tough sentences as a deterrent. Defunding the police puts communities at risk.",
    ),
    (
        "education",
        "Public education should be fully funded and free college should be available to all. Student loan debt should be forgiven. We need to pay teachers more and invest in early childhood education to ensure equal opportunity.",
        "School choice and voucher programs empower parents and improve education through competition. The federal government should have less control over education. Universities have become ideologically biased and need more viewpoint diversity.",
    ),
    (
        "welfare",
        "A strong social safety net is essential for a just society. Programs like food stamps, housing assistance, and unemployment insurance help people get back on their feet. We should expand these programs to reduce poverty and inequality.",
        "Excessive welfare programs create dependency and discourage work. Government handouts trap people in cycles of poverty. We should reform welfare to emphasize work requirements and personal responsibility rather than expanding entitlements.",
    ),
    (
        "environment",
        "Strong environmental regulations are necessary to protect our air, water, and natural resources. Corporations must be held accountable for pollution. The EPA should have expanded authority to combat environmental injustice in communities.",
        "Excessive environmental regulations strangle business growth and cost American jobs. The government should reduce bureaucratic red tape and let the market find efficient solutions. Property rights should take precedence over federal overreach.",
    ),
    (
        "lgbtq_rights",
        "LGBTQ individuals deserve full equal rights including marriage equality and protection from discrimination. Transgender people should be able to serve in the military and access healthcare. Love is love regardless of gender.",
        "Marriage is a sacred institution between a man and a woman. Parents should have the right to protect their children from inappropriate gender ideology in schools. Religious liberty must be protected from government-forced acceptance.",
    ),
    (
        "trade",
        "Trade agreements should prioritize workers' rights, environmental standards, and fair wages globally. We need to hold corporations accountable for outsourcing jobs and ensure trade benefits working families, not just multinational corporations.",
        "Free trade and open markets create prosperity for all. Tariffs and protectionism raise prices for consumers and invite retaliation. America should negotiate bilateral deals that put American businesses and economic freedom first.",
    ),
    (
        "foreign_policy",
        "America should lead through diplomacy, multilateral cooperation, and international institutions. Military intervention should be a last resort. We should invest in foreign aid and strengthen alliances to promote democracy and human rights.",
        "America must maintain the strongest military in the world to protect our interests. We should put America first and not rely on international organizations that undermine our sovereignty. Peace comes through strength, not appeasement.",
    ),
    (
        "tech_regulation",
        "Big tech companies have too much power and need to be regulated. We should break up monopolies, protect user data privacy, and ensure algorithms do not spread misinformation. Digital platforms should be treated as public utilities.",
        "Government regulation of technology stifles innovation and free speech. Big tech censorship of conservative voices is the real problem. The free market should determine which platforms succeed. Section 230 reform should protect all viewpoints.",
    ),
    (
        "housing",
        "Housing is a human right. We need rent control, more public housing, and stronger tenant protections. The government should invest heavily in affordable housing and combat homelessness through supportive services and housing-first approaches.",
        "Rent control and excessive regulations make the housing crisis worse by discouraging construction. The solution is reducing zoning restrictions and letting the free market build more housing. Government housing projects have historically failed.",
    ),
    (
        "drug_policy",
        "The war on drugs has been a failure that disproportionately harms communities of color. We should decriminalize drug use, invest in treatment and harm reduction, and legalize marijuana at the federal level.",
        "Drug legalization sends the wrong message and increases addiction and crime. We need strong enforcement against drug trafficking and dealers. Communities are safer when drug laws are strictly enforced and substance abuse is not normalized.",
    ),
    (
        "voting_rights",
        "Every citizen should have easy access to vote. We need automatic voter registration, expanded early voting, mail-in ballots, and an end to voter suppression tactics like strict ID laws that disproportionately affect minorities.",
        "Election integrity requires voter ID laws to prevent fraud. Loose voting rules invite abuse and undermine public confidence in elections. Only citizens should vote, and we need to clean up voter rolls and ensure secure elections.",
    ),
    (
        "corporate_regulation",
        "Corporations prioritize profits over people and need stronger oversight. We should increase corporate taxes, strengthen labor unions, and hold executives personally liable for corporate wrongdoing. Citizens United should be overturned.",
        "Over-regulation drives businesses overseas and kills economic growth. We should create a business-friendly environment with lower taxes and fewer regulations. Strong corporations create jobs and prosperity that benefit all Americans.",
    ),
    (
        "energy",
        "We must urgently transition to one hundred percent clean renewable energy. Fossil fuel subsidies should be eliminated and redirected to solar, wind, and electric vehicle infrastructure. This transition will create millions of green jobs.",
        "An all-of-the-above energy strategy is essential for American energy independence. We should expand oil drilling, natural gas production, and nuclear power. Forcing a rapid transition to renewables will raise energy costs for families.",
    ),
    (
        "ubi",
        "Universal basic income would provide a floor of economic security, reduce poverty, and give workers bargaining power. Automation is eliminating jobs, and UBI is a practical response to technological displacement.",
        "Universal basic income destroys the incentive to work and would bankrupt the government. People should earn their living through employment. Instead of handouts, we should invest in job training and education.",
    ),
    (
        "medicare_for_all",
        "Medicare for All would cover every American, eliminate insurance company profits, and reduce overall healthcare costs. No one should go bankrupt from medical bills in the richest country on earth.",
        "Medicare for All would eliminate private insurance, raise taxes dramatically, and lead to rationed care. Americans should have the freedom to choose their own healthcare plans, not be forced into a government-run system.",
    ),
    (
        "defund_police",
        "We should redirect police funding to social services, mental health professionals, and community programs that address the root causes of crime. Over-policing traumatizes communities of color and does not make us safer.",
        "Defunding the police is reckless and endangers public safety. Police need more funding and better training, not budget cuts. Crime will surge in communities that reduce law enforcement presence.",
    ),
    (
        "reparations",
        "Reparations for slavery and systemic racism are a moral imperative. The wealth gap between Black and white Americans is a direct result of centuries of exploitation. Financial compensation and investment are needed to begin healing.",
        "Reparations are impractical and unfair. No one alive today was a slaveholder or slave. Dividing Americans by race and forcing some to pay others based on ancestry would deepen racial tensions, not heal them.",
    ),
    (
        "affirmative_action",
        "Affirmative action is necessary to correct systemic inequalities and ensure diversity in education and the workplace. Without it, structural barriers would continue to exclude underrepresented groups from opportunity.",
        "Affirmative action is reverse discrimination that judges people by race rather than merit. Everyone should be evaluated on their qualifications alone. Race-based preferences are unconstitutional and unfair to all.",
    ),
    (
        "trans_sports",
        "Transgender athletes should be allowed to compete in the category that matches their gender identity. Excluding them is discriminatory and harmful. Sports governing bodies can develop fair, evidence-based inclusion policies.",
        "Biological males have inherent physical advantages over females. Allowing transgender women in women's sports is unfair and threatens opportunities for female athletes. Competitive fairness must be protected.",
    ),
    (
        "school_gender",
        "Schools should teach about gender identity and LGBTQ history to promote understanding and protect vulnerable students. Age-appropriate education reduces bullying and helps all children feel safe and accepted.",
        "Parents, not schools, should decide when and how to discuss gender and sexuality with their children. Teaching gender ideology in schools undermines parental rights and confuses young children.",
    ),
    (
        "income_inequality",
        "Extreme income inequality undermines democracy and economic stability. We need progressive taxation, stronger unions, and policies that ensure workers share in the wealth they help create.",
        "Income inequality reflects differences in effort, talent, and risk-taking. Redistribution punishes success and discourages innovation. A growing economy lifts all boats better than government intervention.",
    ),
    (
        "labor_unions",
        "Labor unions are essential for protecting workers' rights, ensuring fair wages, and balancing corporate power. Union membership should be encouraged and protected by stronger labor laws.",
        "Labor unions increase costs for businesses, reduce flexibility, and protect underperforming workers. Right-to-work laws give employees freedom to choose. The free market, not unions, should set wages.",
    ),
    (
        "student_debt",
        "Student loan debt is crushing a generation. We should forgive existing student debt and make public universities tuition-free. Education is a public good that benefits all of society.",
        "Student loan forgiveness is unfair to those who paid their debts or chose not to attend college. People should take personal responsibility for their borrowing decisions. Taxpayers should not subsidize others' degrees.",
    ),
    (
        "campaign_finance",
        "Big money in politics corrupts democracy. We need to overturn Citizens United, impose strict campaign finance limits, and move toward publicly funded elections to restore government by and for the people.",
        "Campaign contributions are a form of free speech protected by the First Amendment. Limiting political spending restricts liberty. Transparency, not government restrictions, is the answer to concerns about political money.",
    ),
    (
        "antitrust",
        "Monopolies harm consumers, workers, and innovation. We need aggressive antitrust enforcement to break up dominant corporations and restore competitive markets that serve the public interest.",
        "Large companies achieve scale because they serve customers well. Breaking them up punishes success and reduces efficiency. Light-touch regulation and market competition, not government intervention, protect consumers.",
    ),
    (
        "free_speech",
        "Free speech has limits. Hate speech, misinformation, and incitement cause real harm and should be regulated on platforms and in public discourse. Protecting vulnerable communities is more important than absolute speech rights.",
        "Free speech is an absolute right. Government or corporate censorship of any viewpoint, no matter how offensive, is a dangerous precedent. The answer to bad speech is more speech, not suppression.",
    ),
    (
        "political_correctness",
        "Using inclusive language and being mindful of how words affect marginalized groups is basic respect. Political correctness helps create a more equitable society and reduces harm to vulnerable people.",
        "Political correctness has gone too far and stifles honest debate. People are afraid to speak their minds. Free expression, even if sometimes offensive, is essential to a healthy democracy and intellectual freedom.",
    ),
    (
        "socialism_capitalism",
        "Democratic socialism provides a more just and equitable society. Essential services like healthcare, education, and housing should not be driven by profit. Scandinavian countries show that socialism and prosperity coexist.",
        "Capitalism is the greatest engine of prosperity and innovation in human history. Socialism leads to government overreach, economic stagnation, and loss of individual freedom. Free markets empower individuals.",
    ),
    (
        "court_packing",
        "Expanding the Supreme Court is necessary to restore balance after Republicans' manipulation of the confirmation process. The court should reflect the will of the people, not be dominated by one political faction.",
        "Court packing is a dangerous assault on judicial independence. The number of justices has been nine since 1869 for good reason. Expanding the court for political advantage would destroy the legitimacy of the judiciary.",
    ),
    (
        "electoral_college",
        "The Electoral College is undemocratic and should be abolished. Every vote should count equally. The president should be elected by national popular vote, not by an outdated system that overrepresents small states.",
        "The Electoral College protects federalism and ensures that candidates campaign across diverse regions. Without it, a few large cities would dominate elections. It preserves the voice of smaller states.",
    ),
    (
        "term_limits",
        "Congressional term limits would reduce corruption, bring fresh perspectives, and end career politicians who prioritize reelection over governing. Democracy benefits from regular turnover in leadership.",
        "Term limits would remove experienced legislators and empower unelected staffers and lobbyists. Voters already have the power to replace their representatives. Experience in government is an asset, not a liability.",
    ),
    (
        "minority_gun_rights",
        "Gun control laws have historically been used to disarm communities of color. The right to bear arms should be protected for all Americans, especially minorities who face threats from both crime and systemic racism.",
        "The Second Amendment applies equally to everyone. However, the best way to protect minority communities is through strong law enforcement and community programs, not by encouraging more firearms in high-crime areas.",
    ),
    (
        "wealth_tax",
        "A wealth tax on the ultra-rich would reduce extreme inequality, fund public services, and ensure billionaires pay their fair share. Concentration of wealth at the top is a threat to democracy.",
        "A wealth tax is unconstitutional, impractical, and would drive capital flight. Taxing unrealized gains punishes investment and entrepreneurship. We should simplify the tax code, not add new taxes that distort markets.",
    ),
    (
        "childcare",
        "Universal childcare is essential for working families and gender equality. The government should fund affordable, high-quality childcare so that no parent has to choose between their career and their children.",
        "Childcare decisions should be left to families, not the government. Federal childcare programs are expensive, one-size-fits-all, and undermine parental choice. Tax credits and flexible work policies are better solutions.",
    ),
    (
        "green_subsidies",
        "Government subsidies for renewable energy are essential investments in our future. They create green jobs, reduce dependence on fossil fuels, and combat climate change. The transition to clean energy needs public support.",
        "Green energy subsidies distort markets, waste taxpayer money, and benefit wealthy companies. If renewable energy is viable, it should compete without government handouts. Subsidies should be ended, not expanded.",
    ),
    (
        "fracking",
        "Fracking poisons water, causes earthquakes, and accelerates climate change. We should ban fracking and transition to clean energy sources. Communities near fracking sites suffer serious health consequences.",
        "Fracking has made America energy independent, created millions of jobs, and lowered energy costs. It is safe when properly regulated. Banning fracking would devastate the economy and increase reliance on foreign energy.",
    ),
    (
        "border_security",
        "A humane immigration system does not require a militarized border. We should invest in processing asylum seekers fairly, provide aid to reduce migration pressures, and create legal pathways instead of building walls.",
        "Securing our borders is the first duty of a sovereign nation. Physical barriers, technology, and more border agents are essential. Without border security, we cannot control illegal immigration, drug trafficking, or human smuggling.",
    ),
    (
        "refugee_admission",
        "America has a moral obligation to welcome refugees fleeing war and persecution. Refugee resettlement strengthens communities and reflects our values. We should increase refugee admissions and expedite processing.",
        "Refugee admission must be strictly limited and carefully vetted to protect national security. We cannot help the world if we cannot take care of our own citizens first. Charity begins at home.",
    ),
    (
        "nationalism_globalism",
        "Global cooperation is essential for addressing shared challenges like climate change, pandemics, and inequality. Nationalism leads to conflict and isolationism. We are stronger when we work together internationally.",
        "National sovereignty must come first. Globalism erodes borders, cultures, and democratic self-governance. Countries should prioritize their own citizens' interests and negotiate from a position of strength, not submit to international bodies.",
    ),
    (
        "qualified_immunity",
        "Qualified immunity shields police officers from accountability for misconduct. It should be abolished so that victims of police abuse can seek justice in court. No one should be above the law.",
        "Qualified immunity is necessary to protect officers from frivolous lawsuits that would make policing impossible. Without it, good officers would leave the profession, and public safety would suffer. Reform, not abolition, is the answer.",
    ),
    (
        "pandemic_response",
        "Strong government intervention during pandemics saves lives. Mask mandates, vaccine requirements, and lockdowns are necessary public health measures. Individual freedom does not include the right to endanger others.",
        "Government overreach during the pandemic destroyed businesses, harmed children's education, and violated civil liberties. Individuals should make their own health decisions. Lockdowns and mandates caused more harm than good.",
    ),
    (
        "crypto_regulation",
        "Cryptocurrency needs strong regulation to protect consumers from fraud, prevent money laundering, and ensure financial stability. The unregulated crypto market is a Wild West that harms ordinary investors.",
        "Over-regulating cryptocurrency stifles innovation and financial freedom. Blockchain technology empowers individuals to control their own money. Light-touch regulation preserves innovation while addressing legitimate concerns.",
    ),
]
