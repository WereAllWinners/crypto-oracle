"""
Dataset Quality Analyzer
Checks completeness, diversity, and quality of training data
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm
import numpy as np

def analyze_dataset(filepath: str):
    """Comprehensive dataset quality analysis"""
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: {filepath}")
    print(f"{'='*70}\n")
    
    examples = []
    with open(filepath, 'r') as f:
        for line in tqdm(f, desc="Loading examples"):
            examples.append(json.loads(line))
    
    total = len(examples)
    print(f"📊 Total examples: {total:,}\n")
    
    # 1. Data Completeness
    print("="*70)
    print("1. DATA COMPLETENESS")
    print("="*70)
    
    has_sentiment = sum(1 for e in examples if e.get('metadata', {}).get('has_sentiment', False))
    has_macro = sum(1 for e in examples if e.get('metadata', {}).get('has_macro', False))
    has_onchain = sum(1 for e in examples if e.get('metadata', {}).get('has_onchain', False))
    has_strategy = sum(1 for e in examples if e.get('metadata', {}).get('has_strategy', False))
    
    print(f"✅ Has Sentiment:  {has_sentiment:,} ({has_sentiment/total*100:.1f}%)")
    print(f"✅ Has Macro:      {has_macro:,} ({has_macro/total*100:.1f}%)")
    print(f"✅ Has On-Chain:   {has_onchain:,} ({has_onchain/total*100:.1f}%)")
    print(f"✅ Has Strategy:   {has_strategy:,} ({has_strategy/total*100:.1f}%)")
    
    # Full signal coverage
    full_coverage = sum(1 for e in examples if all([
        e.get('metadata', {}).get('has_sentiment', False),
        e.get('metadata', {}).get('has_macro', False),
        e.get('metadata', {}).get('has_onchain', False),
        e.get('metadata', {}).get('has_strategy', False)
    ]))
    
    print(f"\n🎯 Full Coverage (all 4 signals): {full_coverage:,} ({full_coverage/total*100:.1f}%)")
    
    # 2. Trading Recommendations Distribution
    print("\n" + "="*70)
    print("2. TRADING RECOMMENDATIONS")
    print("="*70)
    
    recommendations = []
    for e in examples:
        output = e.get('output', '')
        if 'BUY' in output.upper():
            rec = 'BUY'
        elif 'SELL' in output.upper():
            rec = 'SELL'
        elif 'HOLD' in output.upper() or 'NO TRADE' in output.upper():
            rec = 'HOLD'
        else:
            rec = 'UNKNOWN'
        recommendations.append(rec)
    
    rec_counts = Counter(recommendations)
    for rec, count in rec_counts.most_common():
        print(f"{rec:10s}: {count:,} ({count/total*100:.1f}%)")
    
    # 3. Confidence Levels
    print("\n" + "="*70)
    print("3. CONFIDENCE LEVELS")
    print("="*70)
    
    confidences = []
    for e in examples:
        output = e.get('output', '')
        # Extract confidence (looks for patterns like "Confidence: 65%" or "**Confidence: 65%**")
        import re
        match = re.search(r'[Cc]onfidence:?\s*(\d+)%', output)
        if match:
            confidences.append(int(match.group(1)))
    
    if confidences:
        print(f"Average Confidence: {np.mean(confidences):.1f}%")
        print(f"Median Confidence:  {np.median(confidences):.1f}%")
        print(f"Min Confidence:     {min(confidences)}%")
        print(f"Max Confidence:     {max(confidences)}%")
        print(f"Std Dev:            {np.std(confidences):.1f}%")
    
    # 4. Text Length Analysis
    print("\n" + "="*70)
    print("4. TEXT LENGTH ANALYSIS")
    print("="*70)
    
    instruction_lengths = [len(e.get('instruction', '')) for e in examples]
    output_lengths = [len(e.get('output', '')) for e in examples]
    
    print(f"\nInstruction lengths:")
    print(f"  Average: {np.mean(instruction_lengths):,.0f} chars")
    print(f"  Median:  {np.median(instruction_lengths):,.0f} chars")
    print(f"  Min:     {min(instruction_lengths):,} chars")
    print(f"  Max:     {max(instruction_lengths):,} chars")
    
    print(f"\nOutput lengths:")
    print(f"  Average: {np.mean(output_lengths):,.0f} chars")
    print(f"  Median:  {np.median(output_lengths):,.0f} chars")
    print(f"  Min:     {min(output_lengths):,} chars")
    print(f"  Max:     {max(output_lengths):,} chars")
    
    # 5. Pair Distribution
    print("\n" + "="*70)
    print("5. TRADING PAIR DISTRIBUTION")
    print("="*70)
    
    pairs = [e.get('metadata', {}).get('pair', 'UNKNOWN') for e in examples]
    pair_counts = Counter(pairs)
    
    for pair, count in pair_counts.most_common():
        print(f"{pair:12s}: {count:,} ({count/total*100:.1f}%)")
    
    # 6. Temporal Distribution
    print("\n" + "="*70)
    print("6. TEMPORAL DISTRIBUTION")
    print("="*70)
    
    timestamps = []
    for e in examples:
        ts = e.get('metadata', {}).get('timestamp')
        if ts:
            try:
                timestamps.append(pd.to_datetime(ts))
            except:
                pass
    
    if timestamps:
        df_time = pd.DataFrame({'timestamp': timestamps})
        print(f"Date Range: {min(timestamps).date()} to {max(timestamps).date()}")
        print(f"\nExamples per year:")
        yearly = df_time.groupby(df_time['timestamp'].dt.year).size()
        for year, count in yearly.items():
            print(f"  {year}: {count:,} ({count/total*100:.1f}%)")
    
    # 7. Outcome Analysis
    print("\n" + "="*70)
    print("7. OUTCOME ANALYSIS (Actual Results)")
    print("="*70)
    
    profitable = sum(1 for e in examples if e.get('metadata', {}).get('outcome', {}).get('profitable', False))
    unprofitable = total - profitable
    
    print(f"Profitable outcomes: {profitable:,} ({profitable/total*100:.1f}%)")
    print(f"Unprofitable outcomes: {unprofitable:,} ({unprofitable/total*100:.1f}%)")
    
    # Average returns
    returns = [e.get('metadata', {}).get('outcome', {}).get('final_return', 0) for e in examples if e.get('metadata', {}).get('outcome')]
    if returns:
        print(f"\nAverage return: {np.mean(returns):.2f}%")
        print(f"Median return:  {np.median(returns):.2f}%")
        print(f"Best return:    {max(returns):.2f}%")
        print(f"Worst return:   {min(returns):.2f}%")
    
    # 8. Duplicate Detection
    print("\n" + "="*70)
    print("8. DUPLICATE DETECTION")
    print("="*70)
    
    instruction_hashes = [hash(e.get('instruction', '')) for e in examples]
    unique_instructions = len(set(instruction_hashes))
    duplicates = total - unique_instructions
    
    print(f"Unique examples: {unique_instructions:,}")
    print(f"Duplicates: {duplicates:,} ({duplicates/total*100:.1f}%)")
    
    # 9. Quality Score
    print("\n" + "="*70)
    print("9. OVERALL QUALITY SCORE")
    print("="*70)
    
    quality_metrics = {
        'Full Signal Coverage': full_coverage/total,
        'Balanced Recommendations': 1 - abs(0.5 - rec_counts['BUY']/total),  # Closer to 50/50 is better
        'No Duplicates': unique_instructions/total,
        'Temporal Diversity': len(set([t.year for t in timestamps])) / 10 if timestamps else 0,  # More years = better
        'Pair Diversity': len(pair_counts) / 20,  # More pairs = better
    }
    
    overall_score = np.mean(list(quality_metrics.values())) * 100
    
    print(f"\nQuality Metrics:")
    for metric, score in quality_metrics.items():
        print(f"  {metric:25s}: {score*100:5.1f}%")
    
    print(f"\n🎯 Overall Quality Score: {overall_score:.1f}/100")
    
    # 10. Recommendations
    print("\n" + "="*70)
    print("10. RECOMMENDATIONS")
    print("="*70)
    
    issues = []
    
    if full_coverage < total * 0.95:
        issues.append(f"⚠️  Only {full_coverage/total*100:.1f}% have full signal coverage")
    
    if duplicates > total * 0.05:
        issues.append(f"⚠️  {duplicates/total*100:.1f}% duplicates detected")
    
    if len(pair_counts) < 5:
        issues.append(f"⚠️  Limited pair diversity ({len(pair_counts)} pairs)")
    
    # Check if recommendations are too skewed
    max_rec_pct = max(rec_counts.values()) / total
    if max_rec_pct > 0.7:
        issues.append(f"⚠️  Recommendations heavily skewed ({max(rec_counts, key=rec_counts.get)}: {max_rec_pct*100:.1f}%)")
    
    if issues:
        print("\n❌ Issues Found:")
        for issue in issues:
            print(f"   {issue}")
        print(f"\n💡 Recommendation: Consider filtering or sampling the dataset")
    else:
        print("\n✅ Dataset quality is EXCELLENT!")
        print("   All quality checks passed. Safe to use full dataset.")
    
    # Return summary
    return {
        'total': total,
        'full_coverage': full_coverage,
        'quality_score': overall_score,
        'issues': issues
    }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = 'datasets/enhanced_sft_train.jsonl'
    
    analyze_dataset(filepath)