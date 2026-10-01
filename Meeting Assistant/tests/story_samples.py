"""Hand-written interview stories shared by the story-repeat tests.

The MIGRATION_* stories are one story told four ways; every other story is a different one, several of them
on the same theme or in the same business area, which is what makes telling them apart hard."""

LED_MIGRATION = (
    "A while back at my last company we had to move our billing system onto a new platform, and the deadline was only six weeks away. I was asked to lead a team of about six people, so I broke the migration into small pieces with clear owners. Midway through we found that some customer records didn't map cleanly, which put us two weeks behind. I stayed calm, brought the finance team in early, and we rebuilt the mapping together over a long weekend. We went live on time with no customer-facing errors, and I learned that bringing the right people in early beats fixing everything yourself."
)
CONFLICT_SAME_MIGRATION = (
    "Last year, when we were moving the billing system to a new platform, I led a six-person team and we had about six weeks to get it done. Early on a couple of us disagreed about whether to rebuild the customer record mapping or patch it, and it was slowing everyone down. I pulled the two of them aside, listened to both sides, and we agreed to bring finance in and rebuild the mapping together over a weekend. We launched on schedule without any customer errors, and I learned that most disagreements fade once everyone sees the same data."
)
MIGRATION_REWORDED = (
    "At my previous job we were replacing the old invoicing tool with a new platform on a tight timeline, and I ended up coordinating about half a dozen engineers. The trickiest part was that the client data wouldn't line up between the two systems, so we got behind. I looped in the finance folks, we spent a weekend redoing how the records mapped, and we still hit the launch date. I came away convinced that pulling the right people in early saves a lot of pain."
)
MIGRATION_SHORT = (
    "At my last company we had to migrate the customer billing database to a new platform with only six weeks to go. I took the lead of a group of six, split the migration into small pieces with owners, and when some customer records refused to map I brought in finance and we rebuilt the mapping over a weekend. We went live on time, and it taught me to bring the right people in early."
)
HACKATHON = (
    "In my second year I volunteered to run our team's first hackathon, which about twenty colleagues joined. I set up the schedule, matched newer people with experienced ones, and made sure each group had a clear goal by lunchtime. One group was stuck on a data problem, so I connected them with an engineer from another department who unblocked them in an hour. Three of the projects ended up shipping, and I realised that leading is mostly clearing obstacles for others."
)
BILLING_MISTAKE = (
    "Early on, I pushed an invoice change on a Friday afternoon without a second review, and it double-charged a few dozen customers. I told my manager right away, wrote a short apology, and spent the evening reversing the charges. Afterwards I added a checklist and a peer review step for anything touching billing, and we haven't had a repeat. It taught me that speed isn't worth skipping a second pair of eyes."
)
DASHBOARD = (
    "When I joined my last company I inherited a reporting dashboard nobody trusted, because the numbers changed depending on who ran it. I spent the first month interviewing the people who used it, then traced the discrepancies to two spreadsheets feeding the same chart. I merged them into one source and added a short note on how each figure was calculated. Within a quarter the leadership team was using it in weekly reviews, and I saw how much trust comes from simply explaining where numbers come from."
)
TICKETING_ROLLOUT = (
    "When our company switched from one ticketing tool to another, I volunteered to coordinate the rollout for the support group, about fifteen agents. I ran short training sessions, wrote a one-page cheat sheet and set up a chat channel for questions. A few agents resisted at first, so I paired each of them with a fast adopter. By the end of the month ticket handling times were back to normal and the team asked to keep the pairing."
)
CHECKOUT_TIMEOUTS = (
    "Our checkout page kept timing out on mobile, and I was asked to find out why. I dug through the logs, found that an image carousel was blocking the main thread, and worked with a designer to lazy-load it. Conversions went up by about eight percent the next month, and it reminded me to look at real user data before guessing."
)
MIGRATION_AGAIN = (
    "Back when I was at my first job, our customer billing platform was being replaced and I got put in charge of the data side, with a group of about six. We only had a six week window, and halfway through the customer records stopped lining up. I called in someone from finance, and over a weekend we rebuilt how the records were mapped. We went live right on schedule, and I learned to ask for help earlier."
)
SUPPORT_QUEUE = (
    "In my last role our support queue was swamped after a product launch, and I took charge of a small group of five to clear it. I sorted the backlog by urgency, wrote reply templates for the most common problems, and set up a daily ten minute check-in. Within two weeks the queue was back to normal, and the customers who wrote in afterwards said the replies felt personal."
)

MOTIVATION = (
    "I'm really drawn to this role because it combines the analytical work I enjoy with a team that clearly cares "
    "about quality. I think my background in testing gives me a good eye for detail, and I'd love to grow into "
    "mentoring others."
)
MOTIVATION_AGAIN = (
    "What draws me to this role is the mix of analytical work and a team that really cares about quality. My testing "
    "background gives me a good eye for detail, and over time I'd like to grow into mentoring others."
)

SAME_STORY = [LED_MIGRATION, CONFLICT_SAME_MIGRATION, MIGRATION_REWORDED, MIGRATION_SHORT, MIGRATION_AGAIN]
OTHER_STORIES = [HACKATHON, BILLING_MISTAKE, DASHBOARD, TICKETING_ROLLOUT, CHECKOUT_TIMEOUTS, SUPPORT_QUEUE]
