# Exercise 01: Step-by-Step Guide

## Overview

This guide walks you through completing your AI Infrastructure career assessment and planning exercise.

**Estimated Time:** 3-4 hours
**Difficulty:** ⭐ Easy (No coding required)

---

## Step 1: Complete Skills Assessment (45-60 min)

### 1.1 Open the Template

```bash
cd docs/
cp skills-assessment.md skills-assessment-completed.md
```

### 1.2 Rate Each Skill Honestly

Open `skills-assessment-completed.md` in your text editor and rate yourself (0-5) for each skill:

- **0** = Never heard of it
- **1** = Heard of it, read about it
- **2** = Completed tutorials
- **3** = Used in personal projects
- **4** = Used professionally
- **5** = Expert (can teach others)

**Tips:**
- Be brutally honest - this is for you
- Don't inflate scores - it won't help your learning
- It's okay to have many low scores as a beginner

### 1.3 Calculate Your Totals

Add up scores for each category and calculate percentages:

```
Category Score = (Your Score / Max Score) × 100
```

### 1.4 Identify Top 5 Priority Areas

List your 5 lowest-scoring areas that are essential for junior roles:
1. Must focus on these first
2. Find specific resources for each
3. Set measurable goals

**Example:**
- Docker basics: Current 0/5, Target 3/5 in 2 months
- Resource: Docker Mastery course on Udemy

---

## Step 2: Research Job Market (60-90 min)

###2.2 Search for Jobs

Visit these sites and search "Junior AI Infrastructure Engineer":
- [LinkedIn Jobs](https://www.linkedin.com/jobs/)
- [Indeed](https://www.indeed.com/)
- [Glassdoor](https://www.glassdoor.com/)
- [AngelList](https://angel.co/) (startups)

### 2.3 Save 10 Job Descriptions

For each job posting, note:
- Company name
- Job title
- Required skills (hard requirements)
- Preferred skills (nice-to-haves)
- Salary range (if listed)
- Location/remote option

### 2.4 Identify Patterns

Create `docs/job-market-research.md`:

```markdown
# Job Market Research

## Common Requirements
- Python (10/10 postings)
- Docker (9/10 postings)
- Kubernetes (8/10 postings)
- ...

## Common Nice-to-Haves
- Terraform (6/10 postings)
- ML frameworks (7/10 postings)
- ...

## Salary Ranges
- Average: $90K-$110K
- Range: $80K-$130K
- Location: San Francisco adds +$20K

## Company Types
- Big Tech: 3
- Startups: 4
- Enterprise: 3
```

---

## Step 3: Create Learning Plan (60-90 min)

### 3.1 Copy the Template

```bash
cp learning-plan-template.md learning-plan.md
```

### 3.2 Personalize Your Plan

Fill in:
- Your name and start date
- Current experience level
- Available study time per week

### 3.3 Customize Monthly Goals

Based on your skills assessment, adjust:
- **Weak areas** → Spend more time
- **Strong areas** → Move faster or skip

**Example Adjustment:**
```markdown
## Month 1: Foundations

Original plan: Python (2 weeks)
Your adjustment: Python (1 week) - I already know basics
Add: Extra week for Docker (identified as gap)
```

### 3.4 Select Resources

For each topic, choose:
- 1 primary course/book
- 1 practice platform
- 1 project idea

Prioritize:
- Free resources when possible
- Hands-on over theory
- Community-recommended resources

### 3.5 Set Weekly Schedule

Define when you'll study:
```markdown
Monday 7-8PM: Course videos
Tuesday 7-9PM: Hands-on practice
Saturday 10AM-1PM: Project work
```

Be realistic - it's better to do 1 hour daily than 10 hours on weekends.

---

## Step 4: Build Career Roadmap (45-60 min)

### 4.1 Copy the Template

```bash
cp career-roadmap-template.md career-roadmap.md
```

### 4.2 Define Current State

Write honestly:
- Your background (education, work)
- Current skills (from assessment)
- Why AI infrastructure interests you

### 4.3 Set 6-Month Goals

**Be Specific:**
❌ "Learn Kubernetes"
✅ "Deploy 5 apps to Kubernetes, understand pods/services/deployments"

**Make It Measurable:**
- 4 GitHub projects
- 500 GitHub contributions
- 2 certifications

### 4.4 Define 1-Year Vision

Where do you want to be?
- Job title: Junior AI Infrastructure Engineer
- Company: Startup or Big Tech
- Salary: $90K-$120K
- Location: Remote or San Francisco

### 4.5 Sketch 3-Year Path

Long-term vision:
- Mid-level engineer
- Specialization (Kubernetes, MLOps, etc.)
- Industry recognition (blog, talks)

### 4.6 Add Milestones

Set checkpoints:
- 3 months: Modules 1-4 done
- 6 months: Job ready, portfolio complete
- 1 year: Employed, delivering value
- 3 years: Mid-level, specialist

---

## Step 5: Join Communities (15-30 min)

### 5.1 Find Your Tribe

Join at least 2 communities:

**Slack/Discord:**
- [MLOps Community](https://mlops.community/)
- [Kubernetes Community](https://slack.k8s.io/)
- [Python Discord](https://discord.gg/python)

**Reddit:**
- [r/MachineLearning](https://reddit.com/r/MachineLearning)
- [r/kubernetes](https://reddit.com/r/kubernetes)
- [r/devops](https://reddit.com/r/devops)

**Twitter/X:**
Follow AI infrastructure influencers

### 5.2 Introduce Yourself

Post in #introductions channel:
```
Hi! I'm [Name], transitioning to AI Infrastructure.
Background: [Your background]
Learning: [What you're studying]
Goal: Junior AI Infra Engineer in 6 months
Excited to connect!
```

### 5.3 Engage Daily

- Read channels for 10-15 min/day
- Ask questions when stuck
- Share your learning journey
- Help others when you can

---

## Step 6: Validate Your Work (5-10 min)

### 6.1 Run Validation Script

```bash
cd scripts/
python validate_exercise01.py
```

### 6.2 Check Output

Expected:
```
✅ Skills Assessment: Complete
✅ Learning Plan: Complete
✅ Career Roadmap: Complete
⚠️  Job Market Research: Optional (not required)

Score: 3/3 required deliverables complete

🎉 Exercise 01 Complete! Ready for Exercise 02.
```

### 6.3 Fix Any Issues

If validation fails:
- Review the error messages
- Complete missing deliverables
- Run validation again

---

## Step 7: Reflect & Commit (10-15 min)

### 7.1 Journal Entry

Create `docs/reflection.md`:

```markdown
# Exercise 01 Reflection

## What I Learned
- AI Infrastructure is about [...]
- My biggest skill gaps are [...]
- I need to focus on [...]

## Surprises
- I didn't realize [...]
- I was surprised that [...]

## Commitments
I commit to:
1. [...]
2. [...]
3. [...]

## Next Steps
Tomorrow I will:
1. Start Exercise 02
2. [...]
```

### 7.2 Share Your Journey (Optional)

Post on LinkedIn/Twitter:
```
Day 1 of my AI Infrastructure journey! 🚀

Just completed career planning:
✅ Skills assessment done
✅ 6-month learning plan created
✅ Career roadmap defined

Goal: Junior AI Infra Engineer in 6 months

Excited for the journey ahead!

#AIInfrastructure #CareerChange #LearningInPublic
```

---

## Troubleshooting

### "I don't know how to rate myself"

**Solution:** Research the skill first
- Watch a 10-min intro video
- Read the Wikipedia article
- Then rate: 0 if still confused, 1 if you understand the concept

### "All my scores are low"

**Solution:** That's normal for beginners!
- Everyone starts somewhere
- Focus on top 5 priorities
- Celebrate progress, not perfection

### "I can't find 10 hours/week"

**Solution:** Start smaller
- Begin with 5 hours/week
- Build the habit first
- Gradually increase time

### "Job postings are overwhelming"

**Solution:** Break it down
- You don't need ALL skills on day 1
- Junior roles expect you to learn
- Focus on must-haves, not nice-to-haves

### "I'm not sure what to specialize in"

**Solution:** Don't worry yet
- First 6 months: Learn foundations
- Try different areas
- Specialization comes with experience

---

## Success Checklist

Before moving to Exercise 02, ensure you have:

- [ ] Completed skills assessment with honest ratings
- [ ] Identified top 5 priority learning areas
- [ ] Created personalized 6-month learning plan
- [ ] Defined career roadmap with specific goals
- [ ] Researched job market and saved examples
- [ ] Joined at least 2 online communities
- [ ] Validation script passes all checks
- [ ] Reflected on your goals and committed to the plan

---

## Next Steps

**Immediate:**
1. Move to [Exercise 02: Development Environment Setup](../exercise-02-dev-environment/)
2. Start your learning plan TODAY (even just 30 minutes)
3. Schedule your first study session

**This Week:**
- Set up development environment (Exercise 02)
- Begin first learning module from your plan
- Engage in online communities daily

**This Month:**
- Complete Modules 001-002
- Build first portfolio project
- Establish daily learning habit

---

## Additional Resources

### Career Planning Tools
- [LinkedIn Salary Insights](https://www.linkedin.com/salary/)
- [Levels.fyi](https://www.levels.fyi/) - Tech salaries
- [Glassdoor](https://www.glassdoor.com/) - Company reviews

### Learning Platforms
- [Coursera](https://www.coursera.org/)
- [Udemy](https://www.udemy.com/)
- [Pluralsight](https://www.pluralsight.com/)
- [LinkedIn Learning](https://www.linkedin.com/learning/)

### Job Boards
- [LinkedIn Jobs](https://www.linkedin.com/jobs/)
- [Indeed](https://www.indeed.com/)
- [Dice](https://www.dice.com/)
- [Hired](https://hired.com/)
- [AngelList](https://angel.co/)

---

**Congratulations on completing Exercise 01! Your AI Infrastructure journey has officially begun. 🎉**

**Remember:** Progress over perfection. Show up every day, even if just for 30 minutes.

**Next:** [Exercise 02: Development Environment Setup](../exercise-02-dev-environment/README.md)
