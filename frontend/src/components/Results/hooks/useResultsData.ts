import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { abTestAPI, resultsAPI } from '@/utils/api';
import type { TimeSeriesResponse } from '@/types';

export interface ResultsTestSummary {
  test_id: string;
  test_name: string;
  status: string;
  primary_metric: string;
  variants: string[];
  total_users: number;
  completion_percentage: number;
  simulation_status?: string;
}

export interface FinancialImpactResponse {
  test_id: string;
  assumed_arpu: number;
  financial_analysis: {
    control_variant: string | null;
    best_variant: string | null;
    best_observed_variant: string | null;
    incremental_revenue: number;
    best_observed_incremental_revenue: number;
    by_variant: Record<string, {
      uplift_percent: number;
      sample_size: number;
      incremental_revenue: number;
      p_value_corrected: number | null;
      significant: boolean;
    }>;
    control_users: number;
    assumptions: {
      arpu: number;
      uses_significance_gate: boolean;
      significance_threshold: number;
    };
  };
  roi_calculation: {
    base_incremental_revenue: number;
    scenarios: Array<{
      estimated_cost: number;
      rollout_share: number;
      incremental_revenue: number;
      roi_percent: number;
    }>;
    note: string;
  };
}

interface UseResultsDataState {
  tests: ResultsTestSummary[];
  selectedTestId: string;
  setSelectedTestId: (testId: string) => void;
  selectedTest: ResultsTestSummary | undefined;
  timeSeriesData: TimeSeriesResponse | null;
  financialImpact: FinancialImpactResponse | null;
  loading: boolean;
  isSimulating: boolean;
  refreshCurrentTest: () => Promise<void>;
}

export const useResultsData = (): UseResultsDataState => {
  const [tests, setTests] = useState<ResultsTestSummary[]>([]);
  const [selectedTestId, setSelectedTestId] = useState<string>('');
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesResponse | null>(null);
  const [financialImpact, setFinancialImpact] = useState<FinancialImpactResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);
  const pollingRef = useRef<number | null>(null);

  const loadTests = useCallback(async (): Promise<ResultsTestSummary[]> => {
    try {
      const response = await abTestAPI.getAllTests();
      const allTests: ResultsTestSummary[] = [
        ...(response.data.active_tests || []),
        ...(response.data.paused_tests || []),
        ...(response.data.completed_tests || []),
      ].map((t: any) => ({
        test_id: t.test_id,
        test_name: t.test_name,
        status: t.status,
        primary_metric: t.primary_metric,
        variants: t.variants,
        total_users: t.total_users,
        completion_percentage: t.completion_percentage,
        simulation_status: t.simulation_status,
      }));

      setTests(allTests);
      return allTests;
    } catch (error) {
      console.error('Ошибка загрузки тестов:', error);
      return [];
    }
  }, []);

  const loadTimeSeriesData = useCallback(async (testId: string, showLoading: boolean = true) => {
    if (showLoading) {
      setLoading(true);
    }

    try {
      const [timeSeriesResponse, financialImpactResponse] = await Promise.all([
        resultsAPI.getTimeSeriesData(testId),
        resultsAPI.getFinancialImpact(testId),
      ]);

      setTimeSeriesData(timeSeriesResponse.data);
      setFinancialImpact(financialImpactResponse.data);
    } catch (error) {
      console.error('Ошибка загрузки результатов:', error);
      setTimeSeriesData(null);
      setFinancialImpact(null);
    } finally {
      if (showLoading) {
        setLoading(false);
      }
    }
  }, []);

  const refreshCurrentTest = useCallback(async () => {
    if (!selectedTestId) return;
    await loadTimeSeriesData(selectedTestId, false);
  }, [loadTimeSeriesData, selectedTestId]);

  useEffect(() => {
    let mounted = true;

    const bootstrap = async () => {
      const allTests = await loadTests();
      if (!mounted) return;

      if (!selectedTestId && allTests.length > 0) {
        const completedTest = allTests.find((t) => t.status === 'completed');
        const fallback = completedTest ?? allTests[0];
        setSelectedTestId(fallback.test_id);
      }
    };

    bootstrap();

    return () => {
      mounted = false;
      if (pollingRef.current) {
        window.clearInterval(pollingRef.current);
      }
    };
  }, [loadTests, selectedTestId]);

  useEffect(() => {
    const currentlySimulating = tests.some((t) => t.simulation_status === 'running');
    setIsSimulating(currentlySimulating);
  }, [tests]);

  useEffect(() => {
    const selectedTest = tests.find((t) => t.test_id === selectedTestId);
    const shouldPollTests = isSimulating || selectedTest?.status === 'active' || selectedTest?.status === 'paused';

    if (!shouldPollTests) {
      return;
    }

    const testsInterval = window.setInterval(() => {
      loadTests();
    }, 5000);

    return () => {
      window.clearInterval(testsInterval);
    };
  }, [isSimulating, loadTests, selectedTestId, tests]);

  useEffect(() => {
    if (pollingRef.current) {
      window.clearInterval(pollingRef.current);
    }

    if (!selectedTestId) {
      return;
    }

    const selectedTest = tests.find((t) => t.test_id === selectedTestId);
    const selectedTestIsRunning =
      selectedTest?.simulation_status === 'running' ||
      selectedTest?.status === 'active' ||
      selectedTest?.status === 'paused';

    if (!selectedTestIsRunning) {
      loadTimeSeriesData(selectedTestId, true);
      return;
    }

    const pollIntervalMs = selectedTest?.simulation_status === 'running' ? 3000 : 7000;

    loadTimeSeriesData(selectedTestId, true);

    pollingRef.current = window.setInterval(() => {
      loadTimeSeriesData(selectedTestId, false);
    }, pollIntervalMs);

    return () => {
      if (pollingRef.current) {
        window.clearInterval(pollingRef.current);
      }
    };
  }, [loadTimeSeriesData, selectedTestId, tests]);

  const selectedTest = useMemo(
    () => tests.find((t) => t.test_id === selectedTestId),
    [selectedTestId, tests],
  );

  return {
    tests,
    selectedTestId,
    setSelectedTestId,
    selectedTest,
    timeSeriesData,
    financialImpact,
    loading,
    isSimulating,
    refreshCurrentTest,
  };
};
