#!/usr/bin/env python3
"""
temporal_data_manager.py

This module manages the time-based loading of data for the fact-checking system,
handling training, validation, and testing data splits based on time ranges.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import logging
from vector_search import FactCheckSearch
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TimeRange:
    """Time range for data selection with year-month start and end points."""
    start_year: int
    start_month: int
    end_year: int
    end_month: int
    
    def __str__(self) -> str:
        return f"{self.start_year}-{self.start_month:02d} to {self.end_year}-{self.end_month:02d}"
    
    def contains(self, year: int, month: int) -> bool:
        """Check if a given year-month is within this time range."""
        if year < self.start_year or year > self.end_year:
            return False
        if year == self.start_year and month < self.start_month:
            return False
        if year == self.end_year and month > self.end_month:
            return False
        return True
    
    def month_tuple(self) -> Tuple[int, int]:
        """Returns year and month as a tuple for comparison."""
        return (self.end_year, self.end_month)
    
def compare_month_tuples(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Compare two (year, month) tuples, return -1 if a<b, 0 if a==b, 1 if a>b."""
    if a[0] < b[0] or (a[0] == b[0] and a[1] < b[1]):
        return -1
    elif a[0] == b[0] and a[1] == b[1]:
        return 0
    else:
        return 1

def get_next_month(year: int, month: int) -> Tuple[int, int]:
    """Get the next month after the given year and month."""
    if month == 12:
        return (year + 1, 1)
    else:
        return (year, month + 1)

def get_months_between(start_year: int, start_month: int, end_year: int, end_month: int) -> List[Tuple[int, int]]:
    """Get a list of all months between start and end, inclusive of start but exclusive of end."""
    months = []
    current_year, current_month = start_year, start_month
    
    while current_year < end_year or (current_year == end_year and current_month < end_month):
        months.append((current_year, current_month))
        current_year, current_month = get_next_month(current_year, current_month)
    
    return months

class TemporalDataManager:
    """
    Manages time-based loading and processing of fact-checking data.
    """
    
    def __init__(self, data_root: str = "data/processed/balanced"):
        """
        Initialize the temporal data manager.
        
        Args:
            data_root: Root directory containing yearly subdirectories with data
        """
        self.data_root = Path(data_root)
        self.available_years = self._find_available_years()
        logger.info(f"Found data for years: {', '.join(map(str, self.available_years))}")
        
    def _find_available_years(self) -> List[int]:
        """Find all available years in the data directory."""
        years = []
        for item in os.listdir(self.data_root):
            year_dir = self.data_root / item
            if year_dir.is_dir() and (year_dir / "full_set.csv").exists():
                try:
                    years.append(int(item))
                except ValueError:
                    continue
        return sorted(years)
    
    def _load_data_for_timerange(self, time_range: TimeRange) -> pd.DataFrame:
        """
        Load data for a specific time range.
        
        Args:
            time_range: TimeRange object specifying the period to load
            
        Returns:
            DataFrame containing all data within the specified time range
        """
        all_dfs = []
        
        for year in range(time_range.start_year, time_range.end_year + 1):
            if year not in self.available_years:
                logger.warning(f"Year {year} not found in available data")
                continue
                
            year_path = self.data_root / str(year) / "full_set.csv"
            df = pd.read_csv(year_path)
            
            # Filter by month for start and end years
            if year == time_range.start_year and year == time_range.end_year:
                df = df[(df['month'] >= time_range.start_month) & 
                         (df['month'] <= time_range.end_month)]
            elif year == time_range.start_year:
                df = df[df['month'] >= time_range.start_month]
            elif year == time_range.end_year:
                df = df[df['month'] <= time_range.end_month]
                
            # Add year column for tracking
            df['year'] = year
            all_dfs.append(df)
            
        if not all_dfs:
            logger.warning(f"No data found for time range {time_range}")
            return pd.DataFrame()
            
        return pd.concat(all_dfs, ignore_index=True)
    
    def _load_data_for_single_month(self, year: int, month: int) -> pd.DataFrame:
        """
        Load data for a specific month.
        
        Args:
            year: Year to load
            month: Month to load
            
        Returns:
            DataFrame containing data for the specified month
        """
        if year not in self.available_years:
            logger.warning(f"Year {year} not found in available data")
            return pd.DataFrame()
            
        year_path = self.data_root / str(year) / "full_set.csv"
        df = pd.read_csv(year_path)
        
        # Filter to just this month
        df = df[df['month'] == month]
        
        # Add year column for tracking
        df['year'] = year
        
        return df
    
    def build_initial_index(self, 
                           index_time_range: TimeRange, 
                           searcher: FactCheckSearch,
                           output_dir: str,
                           batch_size: int = 512) -> None:
        """
        Build the initial vector database index for the specified time range.
        
        Args:
            index_time_range: Time range for building the initial index
            searcher: FactCheckSearch instance to use
            output_dir: Directory to save the index
            batch_size: Batch size for encoding
        """
        logger.info(f"Building initial vector index for {index_time_range}")
        
        # Load data for the index time range
        df = self._load_data_for_timerange(index_time_range)
        
        if df.empty:
            raise ValueError(f"No data available for index time range {index_time_range}")
            
        # Save to temporary CSV for the searcher to use
        temp_csv = Path(output_dir) / "temp_index_data.csv"
        os.makedirs(os.path.dirname(temp_csv), exist_ok=True)
        df.to_csv(temp_csv, index=False)
        
        # Build and save the index
        searcher.build_index(str(temp_csv), batch_size=batch_size)
        searcher.save_index(output_dir)
        
        # Clean up
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
            
        logger.info(f"Initial vector index built with {len(df)} documents and saved to {output_dir}")
    
    def extend_index_with_month(self,
                              year: int,
                              month: int,
                              searcher: FactCheckSearch,
                              output_dir: str,
                              batch_size: int = 512) -> None:
        """
        Extend an existing index with data from a specific month.
        
        Args:
            year: Year of data to add
            month: Month of data to add
            searcher: FactCheckSearch instance with a loaded index
            output_dir: Directory to save the extended index
            batch_size: Batch size for encoding
        """
        logger.info(f"Extending vector index with data from {year}-{month:02d}")
        
        # Load data for the specific month
        df = self._load_data_for_single_month(year, month)
        
        if df.empty:
            logger.warning(f"No data available for {year}-{month:02d}, index not extended")
            return
            
        # Save to temporary CSV for the searcher to use
        temp_csv = Path(output_dir) / f"temp_extend_{year}_{month:02d}.csv"
        df.to_csv(temp_csv, index=False)
        
        # Extend the existing index
        searcher.extend_index(str(temp_csv), batch_size=batch_size)
        searcher.save_index(output_dir)
        
        # Clean up
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
            
        logger.info(f"Vector index extended with {len(df)} documents from {year}-{month:02d} and saved to {output_dir}")
        
    def _find_most_recent_index(self, base_dir: Path, year: int, month: int) -> Tuple[Path, Optional[Tuple[int, int]]]:
        """
        Find the most recent index available before the specified month.
        
        Args:
            base_dir: Base directory containing all indices
            year: Target year 
            month: Target month
            
        Returns:
            Tuple of (index_path, (last_index_year, last_index_month)) or (None, None) if not found
        """
        # Get all index directories
        if not base_dir.exists():
            return None, None
            
        available_indices = []
        index_pattern = re.compile(r'index_up_to_(\d+)_(\d+)')
        
        # Check for initial index
        initial_index_path = base_dir / "initial_index"
        if (initial_index_path / "index.faiss").exists():
            available_indices.append((initial_index_path, (-1, -1)))  # Use (-1, -1) to mark it as initial
        
        # Check for month-specific indices
        for item in os.listdir(base_dir):
            path = base_dir / item
            if not path.is_dir() or not (path / "index.faiss").exists():
                continue
                
            match = index_pattern.match(item)
            if match:
                index_year = int(match.group(1))
                index_month = int(match.group(2))
                
                # Only consider indices before the target month
                if index_year < year or (index_year == year and index_month < month):
                    available_indices.append((path, (index_year, index_month)))
        
        if not available_indices:
            return None, None
            
        # Sort by year and month (most recent first)
        available_indices.sort(key=lambda x: (-1, -1) if x[1] == (-1, -1) else (-x[1][0], -x[1][1]))
        
        return available_indices[0]
    
    def extract_features_with_temporal_awareness(self,
                                               data_time_range: TimeRange,
                                               index_time_range: TimeRange,
                                               searcher: FactCheckSearch,
                                               index_dir: str,
                                               output_file: str,
                                               tokenizer,
                                               nli_model,
                                               device: str,
                                               top_k: int = 5,
                                               compute_features_fn=None,
                                               batch_size: int = 512) -> None:
        """
        Extract features for claims in the data time range, ensuring proper temporal awareness.
        
        For each month in the data range, we:
        1. Find the most recent index available before the current month
        2. Extend it with any missing months up to the previous month
        3. Extract features for the current month's claims using that index
        
        Args:
            data_time_range: Time range for claims to extract features for
            index_time_range: Initial index time range 
            searcher: FactCheckSearch instance to use
            index_dir: Directory to save/load indexes from
            output_file: Path to save the output features
            tokenizer: NLI tokenizer
            nli_model: NLI model
            device: Device to use
            top_k: Number of top results to retrieve
            compute_features_fn: Function to compute features
            batch_size: Batch size for encoding
        """
        from extract_features import compute_features  # Import here to avoid circular imports
        
        if compute_features_fn is None:
            compute_features_fn = compute_features
        
        # Load the data for the specified range
        df = self._load_data_for_timerange(data_time_range)
        
        if df.empty:
            logger.warning(f"No data available for feature extraction in time range {data_time_range}")
            return
            
        # Sort by year and month to process chronologically
        df = df.sort_values(by=['year', 'month'])
        
        # Group by year and month
        grouped = df.groupby(['year', 'month'])
        
        # Create base index directory
        base_index_dir = Path(index_dir)
        os.makedirs(base_index_dir, exist_ok=True)
        
        # Create initial index with the index time range if needed
        initial_index_dir = base_index_dir / "initial_index"
        os.makedirs(initial_index_dir, exist_ok=True)
        
        # Check if initial index exists, if not, create it
        if not (initial_index_dir / "index.faiss").exists():
            logger.info(f"Building initial index for time range {index_time_range}")
            self.build_initial_index(
                index_time_range,
                searcher,
                str(initial_index_dir),
                batch_size
            )
        
        features_list = []
        
        # Process each month's data
        for (year, month), group_df in grouped:
            logger.info(f"Processing claims from {year}-{month:02d}")
            
            # Target index will include data up to the previous month
            prev_month = month - 1
            prev_year = year
            if prev_month < 1:
                prev_month = 12
                prev_year = year - 1
                
            # Directory for the index we want to use (up to previous month)
            target_index_dir = base_index_dir / f"index_up_to_{prev_year}_{prev_month:02d}"
            os.makedirs(target_index_dir, exist_ok=True)
            
            # Check if the exact target index already exists
            if (target_index_dir / "index.faiss").exists():
                logger.info(f"Loading existing index for {prev_year}-{prev_month:02d} from {target_index_dir}")
                searcher.load_index(str(target_index_dir))
            else:
                # Find the most recent available index before the target month
                recent_index_path, recent_index_month = self._find_most_recent_index(
                    base_index_dir, prev_year, prev_month
                )
                
                # If no recent index found, use the initial index
                if recent_index_path is None:
                    logger.info(f"No existing index found, loading initial index")
                    recent_index_path = initial_index_dir
                    recent_index_month = index_time_range.month_tuple()
                elif recent_index_month == (-1, -1):  # Initial index
                    logger.info(f"Using initial index as most recent")
                    recent_index_month = index_time_range.month_tuple()
                else:
                    logger.info(f"Found most recent index up to {recent_index_month[0]}-{recent_index_month[1]:02d}")
                
                # Load the most recent index
                searcher.load_index(str(recent_index_path))
                
                # Determine which months we need to add to extend the index
                if recent_index_month[0] == prev_year and recent_index_month[1] == prev_month:
                    # Already at the target month, nothing to add
                    logger.info(f"Index already up-to-date")
                    months_to_add = []
                else:
                    # Get the next month after the one in the loaded index
                    next_year, next_month = get_next_month(recent_index_month[0], recent_index_month[1])
                    
                    # Get all months between (next_month) and (prev_month), inclusive
                    months_to_add = get_months_between(
                        next_year, next_month,
                        prev_year, prev_month + 1  # +1 to include prev_month
                    )
                
                # Extend the index with each required month
                for add_year, add_month in months_to_add:
                    self.extend_index_with_month(
                        add_year,
                        add_month,
                        searcher,
                        str(target_index_dir),
                        batch_size
                    )
                
                # Save the updated index if months were added
                if months_to_add:
                    logger.info(f"Saving updated index to {target_index_dir}")
                    searcher.save_index(str(target_index_dir))
                elif recent_index_path != target_index_dir:
                    # If we're using an older index but not adding months, still save to the target location
                    logger.info(f"Saving index copy to {target_index_dir}")
                    searcher.save_index(str(target_index_dir))
            
            # Now extract features for the current month's claims
            logger.info(f"Extracting features for {len(group_df)} claims from {year}-{month:02d}")
            
            for _, row in group_df.iterrows():
                claim = row['text']
                label = 1 if str(row['mergedTextualRating']).lower() == 'true' else 0
                
                # Get similar articles
                retrieved = searcher.search(claim, top_k)
                
                # Compute features
                feats = compute_features_fn(claim, retrieved, tokenizer, nli_model, device)
                feats['label'] = label
                feats['year'] = year
                feats['month'] = month
                
                features_list.append(feats)
        
        # Create the final features DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Save to CSV
        features_df.to_csv(output_file, index=False)
        logger.info(f"Features extracted and saved to {output_file}")